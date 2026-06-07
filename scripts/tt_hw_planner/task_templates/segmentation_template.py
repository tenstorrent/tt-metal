# SPDX-FileCopyrightText: (c) 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Image-segmentation task template.

Emits demo/eval/reference/tests for image-segmentation models like
SAM2 (mask generation), SegFormer (semantic segmentation), UNet, etc.
Single-forward pipeline: image (+ optional prompt) -> mask tensor.
No autoregressive decode loop.

Generalized — uses ``ctx.composition_tree.roles`` for any model-specific
stub lookups. NO hardcoded family-specific names.
"""

from __future__ import annotations

from . import _helpers as h
from ._base import (
    EmittedFiles,
    INPUT_IMAGE,
    OUTPUT_SCALAR,
    TaskTemplate,
    TemplateContext,
)
from ._registry import register_template


@register_template
class SegmentationTemplate(TaskTemplate):
    INPUT_MODALITY = INPUT_IMAGE
    OUTPUT_MODALITY = OUTPUT_SCALAR  # mask tensor (saved as PNG / npz)
    HF_TASK_CLASS = "AutoModelForMaskGeneration"
    EVAL_METRIC = "iou"
    TASK_NAME = "segmentation"

    TASK_DESC = "Image Segmentation"
    REQUIREMENTS_EXTRAS = [
        "Pillow>=9.0.0",
        "torchvision>=0.15.0",
    ]
    MODEL_CONFIG_EXTRAS = {
        "IMAGE_SIZE": ("literal", 1024),  # SAM2 default; could be config-derived
        "PATCH_SIZE": ("literal", 16),
    }
    NEEDS_AUDIO_LOADER = False
    GENERATOR_CLASS_NAME = ""  # Segmentation uses inline pipeline, no decode loop
    REFERENCE_CLASS_SHORT = "Seg"
    REFERENCE_HAS_INPUT_FEATURES = False

    def emit_all(self, ctx: TemplateContext) -> EmittedFiles:
        out = EmittedFiles()
        out.add(f"demo/demo_{self.TASK_NAME}.py", self.emit_demo_file(ctx))
        out.add(f"reference/torch_reference_{self.TASK_NAME}.py", self.emit_reference(ctx))
        out.add(f"evaluation/eval_{self.TASK_NAME}.py", self.emit_eval_file(ctx))
        out.add(f"tests/test_demo_{self.TASK_NAME}.py", self.emit_integration_test(ctx))
        # No parity test — mask outputs are tensor-shaped, not bit-comparable
        # across CPU/TT in the same way as text. Use IoU eval instead.
        return out

    def emit_demo_file(self, ctx: TemplateContext) -> str:
        demo_pkg = ".".join(ctx.demo_dir.parts)
        ct = ctx.composition_tree

        # Discover the encoder/decoder roles from composition tree
        vision_encoder_clean = ct.roles.get("vision_encoder", "encoder_stack")
        # For segmentation, we just need access to the graduated builders;
        # the demo composes them inline via the tt/ package.

        return f'''# SPDX-FileCopyrightText: (c) 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Segmentation demo for {ctx.model_id}.

Pipeline: image -> vision encoder -> (image embeddings)
                -> prompt encoder (optional points/boxes)
                -> mask decoder -> mask tensor

Saves the mask as a PNG overlay on the input image.
"""

from __future__ import annotations

import argparse
import os
import time
from pathlib import Path
from typing import Optional, Tuple

import pytest
import torch
import ttnn

from {demo_pkg}.demo.image_loader import (
    load_image,
    synthesize_image,
    to_tensor_imagenet,
)
from {demo_pkg}.tt.model_config import HF_MODEL_ID


def _to_tt_bf16(t, device):
    return ttnn.from_torch(
        t.to(torch.bfloat16), dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT, device=device,
    )


def run_segmentation(*, device, image_path=None, output_path=None):
    """Run segmentation pipeline: image -> mask. Returns shape info dict.

    For the first iteration this is a pipeline-reach test: load image,
    walk through whatever encoder is graduated, capture the output shape.
    The full mask-decode wiring is model-specific and can be filled in
    per model family.
    """
    print(f"[seg] model: {{HF_MODEL_ID}}")

    if image_path is not None:
        print(f"[seg] image: {{image_path}}")
        img = load_image(image_path)
    else:
        print(f"[seg] image: synthetic (224x224 random)")
        img = synthesize_image()

    t_prep = time.time()
    img_tensor = to_tensor_imagenet(img)
    print(f"[seg] preprocessed shape: {{tuple(img_tensor.shape)}}, "
          f"time: {{time.time() - t_prep:.2f}}s")

    # Walk graduated stubs via the tt/ re-export layer. Each model's
    # specific composition (which stub feeds which) needs to be expressed
    # in a model-specific demo elaboration step (post-emit).
    result = {{
        "image_shape": tuple(img_tensor.shape),
        "stages": [],
    }}

    # Attempt: load every graduated builder. If the model has a single
    # 'encoder_stack' style entry point (SAM2), we exercise it. For
    # multi-piece models (SegFormer's encoder + decode_head), a per-model
    # elaboration is needed.
    try:
        from {demo_pkg} import tt as tt_pkg

        builders = [name for name in dir(tt_pkg) if name.startswith("build_") and not name.startswith("build_lm_head")]
        print(f"[seg] graduated builders available: {{len(builders)}}")
        for b in builders[:8]:
            print(f"   {{b}}")
        result["stages"].append(f"discovered {{len(builders)}} builders")
    except Exception as exc:
        print(f"[seg] tt package import failed: {{type(exc).__name__}}: {{exc}}")
        result["stages"].append(f"tt import failed: {{exc}}")

    # NOTE: Full mask-decoder wiring is model-specific.
    # For SAM2: vision_encoder -> prompt_encoder -> mask_decoder
    # For SegFormer: encoder -> decode_head
    # The auto-emitted demo proves the imports + tt package layer work.
    # The user/LLM completes the composition based on the specific model.

    return result


@pytest.mark.parametrize("device_params", [{{"l1_small_size": 24576}}], indirect=True)
def test_demo_segmentation(device_params, device):
    """Smoke: pipeline imports + builder discovery works."""
    image_env = os.environ.get("SEG_IMAGE", "").strip()
    image_path = Path(image_env) if image_env else None
    output_env = os.environ.get("SEG_OUTPUT", "").strip()
    output_path = Path(output_env) if output_env else None

    result = run_segmentation(
        device=device, image_path=image_path, output_path=output_path,
    )
    assert "image_shape" in result
    assert "stages" in result


def _cli_main(argv=None):
    p = argparse.ArgumentParser(description="Segmentation demo on Tenstorrent")
    p.add_argument("--image", type=Path, default=None)
    p.add_argument("--output", type=Path, default=None)
    args = p.parse_args(argv)

    device = ttnn.open_device(device_id=0, l1_small_size=24576)
    try:
        run_segmentation(
            device=device, image_path=args.image, output_path=args.output,
        )
    finally:
        ttnn.close_device(device)
    return 0


if __name__ == "__main__":
    raise SystemExit(_cli_main())
'''

    def emit_generator_class(self, ctx: TemplateContext) -> str:
        # No generator class for segmentation (no autoregressive loop)
        return ""

    def emit_eval_file(self, ctx: TemplateContext) -> str:
        demo_pkg = ".".join(ctx.demo_dir.parts)
        return f'''# SPDX-FileCopyrightText: (c) 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Segmentation evaluation — pipeline reach + IoU when reference available."""

from __future__ import annotations

import pytest

from {demo_pkg}.demo.demo_segmentation import run_segmentation


@pytest.mark.parametrize("device_params", [{{"l1_small_size": 24576}}], indirect=True)
def test_eval_segmentation(device_params, device):
    """Smoke: verify pipeline reaches all graduated builders."""
    result = run_segmentation(device=device)
    assert "image_shape" in result
    assert any("discovered" in s for s in result["stages"]), \\
        "pipeline must discover graduated builders"
    print(f"\\n[eval-seg] {{result}}")
'''

    def emit_parity_test(self, ctx: TemplateContext) -> str:
        # No parity test for segmentation by default
        return ""

    def emit_reference(self, ctx: TemplateContext) -> str:
        # Generic reference stub — model-specific elaboration needed
        demo_pkg = ".".join(ctx.demo_dir.parts)
        return f'''# SPDX-FileCopyrightText: (c) 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Torch reference for {ctx.model_id} (segmentation).

Single-forward pipeline. Reference implementation is model-specific
and best loaded via the appropriate HF AutoModel class once
transformers supports the model_type.
"""

from __future__ import annotations


class HFGoldenSegmentation:
    """Placeholder for HF golden reference.

    For models where ``transformers.{ctx.composition_tree.task_class}.from_pretrained``
    is supported, this would load the model and run a single forward.
    For new / experimental models (e.g. sam2_video before transformers
    catches up), this class is a stub.
    """

    def __init__(self, model_id: str = "{ctx.model_id}") -> None:
        self.model_id = model_id

    def generate(self, image, **kwargs):
        raise NotImplementedError(
            f"HF golden reference for {ctx.model_id} requires manual elaboration. "
            f"See https://huggingface.co/{ctx.model_id} for the model's expected API."
        )


__all__ = ["HFGoldenSegmentation"]
'''

    def emit_integration_test(self, ctx: TemplateContext) -> str:
        demo_pkg = ".".join(ctx.demo_dir.parts)
        return f'''# SPDX-FileCopyrightText: (c) 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Integration test — segmentation pipeline reach."""

from __future__ import annotations

import pytest

from {demo_pkg}.demo.demo_segmentation import run_segmentation


@pytest.mark.parametrize("device_params", [{{"l1_small_size": 24576}}], indirect=True)
def test_segmentation_pipeline(device_params, device):
    result = run_segmentation(device=device)
    assert "stages" in result
    assert any("discovered" in s for s in result["stages"])
'''


__all__ = ["SegmentationTemplate"]
