# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

# SPDX-License-Identifier: Apache-2.0

"""End to end: an image of your choice through TRELLIS.2 (512 by default) with the flow DiTs on one chip, to a PLY.

Needs TRELLIS2_IMAGE (an RGBA image of one object), TRELLIS2_REPO (a TRELLIS.2 75fbf018 clone with
demo/trellis2_cpu_overlay.patch applied) and DINOv3 access on Hugging Face; weights go to TRELLIS2_WEIGHTS_DIR
(default ~/.cache/trellis2_tt). TRELLIS2_PIPELINE_TYPE picks 512 / 1024 / 1024_cascade / 1536_cascade; the output goes
to TRELLIS2_OUT or pytest's tmp dir.
"""

from __future__ import annotations

import os

import pytest

from models.experimental.trellis2.demo import demo


@pytest.mark.skipif(
    not (os.environ.get("TRELLIS2_REPO") and os.environ.get("TRELLIS2_IMAGE")),
    reason="set TRELLIS2_REPO and TRELLIS2_IMAGE (see demo/README.md)",
)
@pytest.mark.parametrize("mesh_device", [(1, 1)], indirect=True)
def test_demo_e2e(mesh_device, tmp_path):
    summary = demo.run_e2e(
        mesh_device,
        trellis_dir=os.environ["TRELLIS2_REPO"],
        weights_dir=os.path.expanduser(os.environ.get("TRELLIS2_WEIGHTS_DIR", "~/.cache/trellis2_tt")),
        image_path=os.environ["TRELLIS2_IMAGE"],
        out_dir=os.environ.get("TRELLIS2_OUT", str(tmp_path)),
        pipeline_type=os.environ.get("TRELLIS2_PIPELINE_TYPE", "512"),
    )
    assert summary["structure_voxels"] > 0
    assert summary["mesh"]["vertices"] > 0 and summary["mesh"]["faces"] > 0
    assert summary["mesh"]["uncoloured_vertices"] == 0
    assert os.path.getsize(summary["ply"]) > 0
