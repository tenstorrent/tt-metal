#!/bin/bash

set -e
echo "Running mochi tests"

if [ -z "${FAKE_DEVICE}" ]; then
    echo "Error: FAKE_DEVICE environment variable must be set"
    exit 1
fi

# pytest models/experimental/mochi/tests/test_mod_rmsnorm.py
pytest models/experimental/mochi/tests/test_residual_tanh_gated_rmsnorm.py
pytest models/experimental/mochi/tests/test_rope.py
pytest models/experimental/mochi/tests/test_tt_embed.py
pytest models/experimental/mochi/tests/test_tt_feedforward.py
pytest models/experimental/mochi/tests/test_tt_final_layer.py
pytest models/experimental/mochi/tests/test_tt_attn.py
pytest models/experimental/mochi/tests/test_tt_block.py::test_tt_block

pytest models/experimental/mochi/tests/test_tt_asymm_dit_joint.py::test_tt_asymm_dit_joint_prepare
pytest models/experimental/mochi/tests/test_tt_asymm_dit_joint.py::test_tt_asymm_dit_joint_inference -k "L1"
pytest models/experimental/mochi/tests/test_tt_asymm_dit_joint.py::test_tt_asymm_dit_joint_model_fn -k "L1"

pytest models/experimental/mochi/tests/test_pipeline.py -k "L1"
