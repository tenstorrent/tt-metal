# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""End-to-end FIBO *product* path: natural-language text -> structured JSON -> image.

Stage A (host CPU): ``briaai/FIBO-vlm`` (a Qwen3-VL model, ~8.9 GB) converts a free-text prompt into
the structured JSON caption FIBO was trained on. It is loaded via the ``briaai/FIBO-VLM-prompt-to-JSON``
modular-pipeline block's ``TransformersEngine`` / ``generate_json_prompt`` helpers *directly* -- the
block's ``__init__`` hardcodes ``.to("cuda")``, which we don't have, so we bypass it and run the VLM on
CPU (bf16).

Stage B (TT, 2x2 Blackhole): the JSON string is fed to ``BriaFiboPipeline`` -> 1024x1024 image, decoded
on-device.

SLOW (host CPU autoregressive decode of the JSON caption on an 8B VLM) -- run on-demand, not fast CI.
"""
import gc
import importlib.util
import json
import os

import pytest
from huggingface_hub import snapshot_download

import ttnn

FIBO_PATH = os.environ.get("FIBO_PATH", "briaai/FIBO")
FIBO_VLM_BLOCKS = "briaai/FIBO-VLM-prompt-to-JSON"  # tiny: the modular-pipeline custom code
FIBO_VLM_MODEL = "briaai/FIBO-vlm"  # the ~8.9 GB Qwen3-VL weights


def _local(repo):
    try:
        return snapshot_download(repo, local_files_only=True)
    except Exception as e:
        pytest.skip(f"{repo} not cached: {e}")


def _load_vlm_module(blocks_dir):
    """Import the FIBO-VLM remote-code module from its cached snapshot (equivalent to trust_remote_code)."""
    path = os.path.join(blocks_dir, "fibo_vlm_prompt_to_json.py")
    spec = importlib.util.spec_from_file_location("fibo_vlm_prompt_to_json", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


@pytest.mark.parametrize("mesh_device", [(2, 2)], indirect=["mesh_device"])
@pytest.mark.parametrize(
    "device_params",
    [{"fabric_config": ttnn.FabricConfig.FABRIC_1D, "l1_small_size": 32768, "trace_region_size": 50000000}],
    indirect=["device_params"],
)
def test_fibo_vlm_to_image_e2e(*, mesh_device):
    import numpy as np

    ckpt = _local(FIBO_PATH)
    blocks_dir = _local(FIBO_VLM_BLOCKS)
    _local(FIBO_VLM_MODEL)  # ensure the VLM weights are cached (skip cleanly otherwise)
    vlm = _load_vlm_module(blocks_dir)

    # --- Stage A: text -> structured JSON on host CPU (bypass the block's hardcoded .to("cuda")). ---
    engine = vlm.TransformersEngine(FIBO_VLM_MODEL)  # Qwen3-VL, bf16, defaults to CPU
    json_prompt = vlm.generate_json_prompt(
        vlm_processor=engine,
        prompt="a luxury sports car on a wet city street at night, neon reflections, cinematic",
        image=None,
        structured_prompt=None,
        top_p=0.9,
        temperature=0.0,  # greedy -> deterministic caption
        max_tokens=2048,
        stop=["<|im_end|>", "<|end_of_text|>"],
    )
    # generate_json_prompt -> prepare_clean_caption -> a minimal JSON string (FIBO's caption format).
    assert isinstance(json_prompt, str) and json_prompt, "VLM did not return a JSON string"
    parsed = json.loads(json_prompt)  # must be valid JSON
    assert isinstance(parsed, dict) and parsed, "VLM JSON is empty / not a JSON object"
    with open("fibo_vlm_prompt.json", "w") as f:
        f.write(json_prompt)

    del engine
    gc.collect()

    # --- Stage B: structured JSON -> image on the 2x2 Blackhole mesh (on-device decode). ---
    from models.tt_dit.pipelines.bria_fibo.pipeline_bria_fibo import BriaFiboPipeline, BriaFiboPipelineConfig

    pipe = BriaFiboPipeline(
        device=mesh_device,
        config=BriaFiboPipelineConfig.default(
            mesh_shape=mesh_device.shape, checkpoint_name=ckpt, height=1024, width=1024
        ),
    )
    imgs = pipe(json_prompt, num_inference_steps=30, guidance_scale=5.0, seed=0, force_device_decode=True)

    arr = np.asarray(imgs[0])
    assert arr.shape == (1024, 1024, 3), f"unexpected image shape {arr.shape}"
    assert arr.std() > 1.0, f"image looks degenerate (std={arr.std():.4f})"
    assert np.unique(arr).size > 16, f"image looks degenerate ({np.unique(arr).size} unique values)"
    imgs[0].save("fibo_vlm_e2e.png")
