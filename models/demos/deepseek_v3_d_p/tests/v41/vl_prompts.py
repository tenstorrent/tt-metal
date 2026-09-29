# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

# SPDX-License-Identifier: Apache-2.0

"""Merged-sequence (image + text) prompts of the V4.1 VL tests (bead 10.2); pure torch, so the CPU oracles can be
precomputed outside the device lock with exactly the prompts the device tests use.

A prompt is real text (``oracle.text_tokens``) with image spans written over some of its positions, all inside
the first chunk (the reference's rule); the scored tail stays text. Image features are teacher-forced aligner
rows: synthetic at small dims, the ``vision_oracle`` rows of a test image at real dims.
"""

from models.demos.deepseek_v3_d_p.reference.deepseek_v41 import oracle as orc

SMALL_SEQ = 512
# (start, n_llm_h, n_llm_w): two images, 6*9+2 = 56 and 4*6+2 = 26 positions, both inside a 256-token first chunk
SMALL_SPANS = ((8, 6, 8), (120, 4, 5))
PRODUCTION_SEQ = 2048
PRODUCTION_IMAGE = (640, 480)  # -> 35x46 patches -> 12x16 aligner rows -> 206 positions
PRODUCTION_IMAGE_START = 32
IMAGE_SEED = 101  # test_vision's seed: the synthetic image and its cached vision oracles are shared


def small_prompt(dim: int, seq: int = SMALL_SEQ, seed: int = 0):
    """(tokens [1, seq], VLPrompt) with synthetic aligner rows of width ``dim``."""
    spans = [(s, h, w, orc.synthetic_image_features(h * w, dim, seed + i)) for i, (s, h, w) in enumerate(SMALL_SPANS)]
    return orc.merged_prompt(orc.text_tokens(seq), spans)


def production_prompt(checkpoint=None, seq: int = PRODUCTION_SEQ):
    """(tokens [1, seq], VLPrompt, (patches, n_h, n_w)): one test image's reference aligner rows (the checkpoint's
    vision tower with ``checkpoint``, else the synthetic one of IMAGE_SEED; cached by ``vision_oracle``)."""
    args = orc.vision_args()
    patches, n_h, n_w = orc.image_patches(orc.synthetic_image(*PRODUCTION_IMAGE, IMAGE_SEED), args)
    vision = orc.vision_oracle(patches, n_h, n_w, args=args, seed=IMAGE_SEED, checkpoint=checkpoint)
    meta = vision["meta"]
    span = (PRODUCTION_IMAGE_START, meta["n_llm_h"], meta["n_llm_w"], vision["aligned"])
    tokens, prompt = orc.merged_prompt(orc.text_tokens(seq), [span])
    return tokens, prompt, (patches, n_h, n_w)
