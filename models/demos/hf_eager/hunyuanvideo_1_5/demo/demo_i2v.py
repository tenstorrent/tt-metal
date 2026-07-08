# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

"""Image-to-video (i2v) denoise-step demo for ``tencent/HunyuanVideo-1.5``.

Runs one real forward of the chained TTNN pipeline in the i2v conditioning
regime (dual text embeddings + an active image embedding; all condition tokens
valid) and prints the denoised velocity prediction + PCC vs the HF golden.

    python -m models.demos.hf_eager.hunyuanvideo_1_5.demo.demo_i2v
"""

from __future__ import annotations

from models.demos.hf_eager.hunyuanvideo_1_5.demo._common import build_argparser, run_demo


def main():
    args = build_argparser("i2v").parse_args()
    run_demo("i2v", args)


if __name__ == "__main__":
    main()
