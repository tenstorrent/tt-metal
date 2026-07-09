# SPDX-FileCopyrightText: © 2024 Tenstorrent USA, Inc.

# SPDX-License-Identifier: Apache-2.0

# The reference loader lives in reference/load_falcon_weights.py so non-test code (the demo)
# can use it without importing from tests/. Re-exported here for the existing test imports.
from models.demos.t3000.falcon40b.reference.load_falcon_weights import load_falcon_reference_model  # noqa: F401


def load_hf_model(model_version):
    hugging_face_reference_model = load_falcon_reference_model(model_version)
    state_dict = hugging_face_reference_model.state_dict()

    return hugging_face_reference_model, state_dict
