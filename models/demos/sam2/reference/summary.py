# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""
Dumps exact op boundaries and tensor hierarchy ([96, 192, 384, 768]) of facebook/sam2-hiera-tiny.
Used as institutional reference summary of block resolutions, channel counts, and tensor bounds.
"""

import sys
from pathlib import Path
import torch

root_dir = Path(__file__).parent.parent
if str(root_dir) not in sys.path:
    sys.path.insert(0, str(root_dir))

from reference.sam2_reference import Sam2ReferenceImageModel

def dump_tensor_hierarchy() -> list[int]:
    """Runs 1024x1024 dummy image through PyTorch reference model and dumps op boundaries/hierarchy."""
    model = Sam2ReferenceImageModel(embed_dim=96)
    model.eval()

    torch.manual_seed(42)
    dummy_input = torch.randn(1, 3, 1024, 1024, dtype=torch.float32)

    with torch.no_grad():
        out = model(dummy_input)

    s1 = out["stage1_features"]
    s2 = out["stage2_features"]
    s3 = out["stage3_features"]
    s4 = out["stage4_features"]

    print("=== SAM 2 (facebook/sam2-hiera-tiny) Op Boundaries & Tensor Hierarchy ===")
    print(f"Input Image        : shape={list(dummy_input.shape)}, dtype={dummy_input.dtype}")
    print(f"Stage 1 (Stride  4): shape={list(s1.shape)}, blocks=1, channels={s1.shape[1]}")
    print(f"Stage 2 (Stride  8): shape={list(s2.shape)}, blocks=2, channels={s2.shape[1]}")
    print(f"Stage 3 (Stride 16): shape={list(s3.shape)}, blocks=7, channels={s3.shape[1]}")
    print(f"Stage 4 (Stride 32): shape={list(s4.shape)}, blocks=2, channels={s4.shape[1]}")
    print("=========================================================================")

    hierarchy = [s1.shape[1], s2.shape[1], s3.shape[1], s4.shape[1]]
    print(f"Tensor Channel Hierarchy: {hierarchy}")
    return hierarchy

if __name__ == "__main__":
    hierarchy = dump_tensor_hierarchy()
    assert hierarchy == [96, 192, 384, 768], f"Expected [96, 192, 384, 768], got {hierarchy}"
