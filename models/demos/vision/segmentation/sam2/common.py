# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI U.S. Corp.
# SPDX-License-Identifier: Apache-2.0

import os

from transformers import AutoProcessor, Sam2VideoModel

MODEL_NAME = "facebook/sam2-hiera-tiny"


def load_sam2_model_and_processor(model_location_generator=None):
    """Load the Hugging Face SAM2 video checkpoint and processor."""
    source = os.environ.get("HF_MODEL")
    if source is None and "TT_GH_CI_INFRA" in os.environ:
        if model_location_generator is None:
            raise ValueError("model_location_generator is required in CI v2")
        source = str(model_location_generator(MODEL_NAME, download_if_ci_v2=True))
    source = source or MODEL_NAME
    model = Sam2VideoModel.from_pretrained(source).eval()
    processor = AutoProcessor.from_pretrained(source)
    return model, processor
