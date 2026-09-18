"""Run the SD3.5-Large 4-chip (2x2, cfg1 sp2 tp2) pipeline on a 2x2 submesh of a Galaxy.

Opening a bare 2x2 mesh on a Galaxy fails fabric router sync (neighbours outside the mesh never
come up), so open the full system mesh with fabric and slice a 2x2 submesh out of it. Mirrors
test_sd35_pipeline[2x2cfg0sp0tp1-yes_traced] otherwise. Env: SD35_STEPS (28), SD35_QUANT (model
code reads it), SD35_TRACED (1).
"""

import os
import sys
import time

sys.path.insert(0, os.getcwd())
from loguru import logger

import ttnn
from conftest import reset_fabric, set_fabric
from models.perf.benchmarking_utils import BenchmarkProfiler
from models.tt_dit.parallel.config import DiTParallelConfig
from models.tt_dit.pipelines.events import profiler_event_callback
from models.tt_dit.pipelines.stable_diffusion_35_large.pipeline_stable_diffusion_35_large import (
    StableDiffusion3Pipeline,
    StableDiffusion3PipelineConfig,
)

steps = int(os.environ.get("SD35_STEPS", "28"))
traced = os.environ.get("SD35_TRACED", "1") == "1"
cfg_enabled = os.environ.get("SD35_CFG", "1") == "1"
# 4-chip layouts. On a Galaxy only the 2x2 corner is a physically closed ring of 4 chips (a native
# 1x4 row hangs in CCLs), so the tp4 / sp4 layouts take the corner and relabel it to 1x4 / 4x1,
# which the reshape does in ring order (device ids 0,1,5,4).
#   "2x2"  = sp2 (axis 0) x tp2 (axis 1)          "1x4c" = tp4 (axis 1) on the reshaped corner
#   "4x1c" = sp4 (axis 0) on the reshaped corner  "1x4"/"4x1" = native row/column (hang on Galaxy)
layout = os.environ.get("SD35_LAYOUT", "4x1tp")
_layouts = {
    "2x2": (ttnn.MeshShape(2, 2), None, (2, 0), (2, 1)),
    "1x4c": (ttnn.MeshShape(2, 2), ttnn.MeshShape(1, 4), (1, 0), (4, 1)),
    "4x1c": (ttnn.MeshShape(2, 2), ttnn.MeshShape(4, 1), (4, 0), (1, 1)),
    "1x4": (ttnn.MeshShape(1, 4), None, (1, 0), (4, 1)),
    "4x1": (ttnn.MeshShape(4, 1), None, (4, 0), (1, 1)),
    # Native column of the Galaxy (a complete system-mesh axis): tp4 along axis 0, no sequence parallelism.
    "4x1tp": (ttnn.MeshShape(4, 1), None, (1, 1), (4, 0)),
}
mesh_shape, reshape_to, sp_cfg, tp_cfg = _layouts[layout]
topology = {"linear": ttnn.Topology.Linear, "ring": ttnn.Topology.Ring}[os.environ.get("SD35_TOPOLOGY", "ring")]
fabric = {"linear": ttnn.FabricConfig.FABRIC_1D, "ring": ttnn.FabricConfig.FABRIC_1D_RING}[
    os.environ.get("SD35_FABRIC", os.environ.get("SD35_TOPOLOGY", "ring"))
]
num_links = int(os.environ.get("SD35_LINKS", "2"))
tag = os.environ.get(
    "SD35_TAG", f"{layout}_{os.environ.get('SD35_TOPOLOGY', 'ring')}_{os.environ.get('SD35_QUANT', 'bf16')}_s{steps}"
)

_mmrs_cfg = os.environ.get("SD35_MMRS_CFG")  # override the ff2 MMRS blocking / handoff for A/B runs
if _mmrs_cfg:
    from models.tt_dit.utils import matmul as _mm

    _gx, _gy, _m, _k, _n, _sh, _sw, _win = _mmrs_cfg.split(",")
    _mm.fused_mmrs_configs[ttnn.CoreCoord(12, 10)][(8192, 2432, 2432)] = _mm.FusedMMRSConfig(
        ttnn.CoreCoord(int(_gx), int(_gy)),
        int(_m),
        int(_k),
        int(_n),
        int(_sh),
        int(_sw),
        None,
        1,
        None if _win == "none" else int(_win),
    )
    logger.info(f"MMRS config override: {_mmrs_cfg}")
set_fabric(fabric)
full = ttnn.open_mesh_device(
    mesh_shape=ttnn.MeshShape(4, 8),
    l1_small_size=int(os.environ.get("SD35_L1_SMALL", "65536")),
    trace_region_size=50_000_000,
)
try:
    sub = full.create_submeshes(mesh_shape)[0]
    if reshape_to is not None:
        sub.reshape(reshape_to)  # permanent relabel (ring order) for this run
    logger.info(
        f"submesh {sub.shape} of {full.shape} layout={layout} sp={sp_cfg} tp={tp_cfg} topology={topology} fabric={fabric} links={num_links}"
    )
    pipeline = StableDiffusion3Pipeline(
        device=sub,
        config=StableDiffusion3PipelineConfig.default(
            mesh_shape=sub.shape,
            dit_parallel_config=DiTParallelConfig.from_tuples(cfg=(1, 0), sp=sp_cfg, tp=tp_cfg),
            topology=topology,
            num_links=num_links,
            width=1024,
            height=1024,
            cfg_enabled=cfg_enabled,
            # Keep T5 off on every layout so 2x2 (where it cannot fit) and 1x4 compare like for like.
            enable_t5_text_encoder=os.environ.get("SD35_T5", "0") == "1",
            checkpoint_name="stabilityai/stable-diffusion-3.5-large",
            vae_spatial=os.environ.get("SD35_VAE_SPATIAL", "1") == "1",
            vae_use_conv3d=os.environ.get("SD35_VAE_CONV3D", "1") == "1",
        ),
    )
    prompt = (
        "An epic, high-definition cinematic shot of a rustic snowy cabin glowing "
        "warmly at dusk, nestled in a serene winter landscape. Surrounded by gentle "
        "snow-covered pines and delicate falling snowflakes - captured in a rich, "
        "atmospheric, wide-angle scene with deep cinematic depth and warmth."
    )
    for it in range(int(os.environ.get("SD35_ITERS", "2"))):
        prof = BenchmarkProfiler()
        with prof("run", iteration=it):
            images = pipeline(
                prompts=[prompt],
                negative_prompts=[""],
                num_inference_steps=steps,
                guidance_scale=3.5 if cfg_enabled else 1.0,
                traced=traced,
                vae_traced=(bool(int(os.environ["SD35_VAE_TRACED"])) if "SD35_VAE_TRACED" in os.environ else None),
                encoder_traced=False,
                on_event=profiler_event_callback(prof, it),
            )
        images[0].save(f"sd35_2x2_{tag}_it{it}.png")
        d = lambda k: prof.get_duration(k, it)
        logger.info(
            f"RESULT tag={tag} layout={layout} topo={os.environ.get('SD35_TOPOLOGY', 'ring')} iter={it} steps={steps} traced={traced} cfg={cfg_enabled} "
            f"encoder={d('encoder'):.2f}s vae={d('vae'):.2f}s denoising={d('denoising'):.2f}s "
            f"step={d('denoising') / steps:.3f}s total={d('total'):.2f}s run={d('run'):.2f}s"
        )
finally:
    for s in full.get_submeshes():
        ttnn.close_mesh_device(s)
    ttnn.close_mesh_device(full)
    reset_fabric(fabric)
print("DRIVER_DONE")
