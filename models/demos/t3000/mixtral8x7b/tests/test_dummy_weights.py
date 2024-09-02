# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0
import os
import pytest
from loguru import logger

import ttnn

from models.demos.t3000.mixtral8x7b.tt.mixtral_model import TtTransformer
from models.demos.t3000.mixtral8x7b.tt.model_config import TtModelArgs
from models.utility_functions import (
    comp_pcc,
    comp_allclose,
)


def test_load_dummy_weights(t3k_mesh_device):
    # Set to incorrect paths to test dummy weight loading

    backup_cache_path = TtModelArgs.DEFAULT_CACHE_PATH
    backup_tokenizer_path = TtModelArgs.DEFAULT_TOKENIZER_PATH
    backup_ckpt_dir = TtModelArgs.DEFAULT_CKPT_DIR

    try:
        TtModelArgs.DEFAULT_CACHE_PATH = "this/path/does/not/exist"
        TtModelArgs.DEFAULT_TOKENIZER_PATH = "this/path/does/not/exist"
        TtModelArgs.DEFAULT_CKPT_DIR = "this/path/does/not/exist"

        model_args = TtModelArgs(t3k_mesh_device.get_device(0), dummy_weights=True)
        model_args.n_layers = 1
        state_dict = model_args.load_state_dict()
        tt_model = TtTransformer(
            mesh_device=t3k_mesh_device,
            state_dict=state_dict,
            args=model_args,
            layers=list(range(model_args.n_layers)),
            dtype=ttnn.bfloat8_b,
        )

        logger.info("Loading dummy weights passed!")
    finally:
        TtModelArgs.DEFAULT_CACHE_PATH = backup_cache_path
        TtModelArgs.DEFAULT_TOKENIZER_PATH = backup_tokenizer_path
        TtModelArgs.DEFAULT_CKPT_DIR = backup_ckpt_dir
