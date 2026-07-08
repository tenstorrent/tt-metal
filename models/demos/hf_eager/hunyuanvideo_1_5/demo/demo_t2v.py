# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

"""Text-to-video (t2v) denoise-step demo for ``tencent/HunyuanVideo-1.5``.

Runs one real forward of the chained TTNN pipeline in the t2v conditioning
regime (dual text embeddings; image conditioning zeroed / masked, matching the
reference ``is_t2v`` path) and prints the denoised velocity prediction + PCC vs
the HF golden.

    python -m models.demos.hf_eager.hunyuanvideo_1_5.demo.demo_t2v
"""

from __future__ import annotations

from models.demos.hf_eager.hunyuanvideo_1_5.demo._common import build_argparser, run_demo


def main():
    args = build_argparser("t2v").parse_args()
    run_demo("t2v", args)


if __name__ == "__main__":
    main()
