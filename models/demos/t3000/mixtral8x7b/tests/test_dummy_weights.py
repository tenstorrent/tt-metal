# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0
import os
import pytest
from loguru import logger

# Set to incorrect paths to test dummy weight loading
os.environ["MIXTRAL_CKPT_DIR"] = "this/path/does/not/exist"
os.environ["MIXTRAL_TOKENIZER_PATH"] = "this/path/does/not/exist"
os.environ["MIXTRAL_CACHE_PATH"] = "this/path/does/not/exist"
os.environ["TT_METAL_ASYNC_DEVICE_QUEUE"] = "1"

import ttnn

from models.demos.t3000.mixtral8x7b.tt.mixtral_model import TtTransformer
from models.demos.t3000.mixtral8x7b.tt.model_config import TtModelArgs
from models.utility_functions import (
    comp_pcc,
    comp_allclose,
)


def test_load_dummy_weights(t3k_device_mesh):
    model_args = TtModelArgs(t3k_device_mesh.get_device(0), dummy_weights=True)
    model_args.n_layers = 1
    state_dict = model_args.load_state_dict()
    tt_model = TtTransformer(
        device_mesh=t3k_device_mesh,
        state_dict=state_dict,
        args=model_args,
        layers=list(range(model_args.n_layers)),
        dtype=ttnn.bfloat8_b,
    )

    logger.info("Loading dummy weights passed!")
