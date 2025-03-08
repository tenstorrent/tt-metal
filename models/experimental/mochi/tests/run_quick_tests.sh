#!/bin/bash

set -e
echo "Running mochi tests"

if [ -z "${FAKE_DEVICE}" ]; then
    echo "Error: FAKE_DEVICE environment variable must be set"
    exit 1
fi

# DiT tests
pytest models/experimental/mochi/tests/dit/test_norms.py
pytest models/experimental/mochi/tests/dit/test_rope.py
pytest models/experimental/mochi/tests/dit/test_embed.py
pytest models/experimental/mochi/tests/dit/test_mlp.py
pytest models/experimental/mochi/tests/dit/test_final_layer.py
pytest models/experimental/mochi/tests/dit/test_attention.py
pytest models/experimental/mochi/tests/dit/test_block.py::test_tt_block

pytest models/experimental/mochi/tests/dit/test_model.py::test_tt_asymm_dit_joint_prepare
pytest models/experimental/mochi/tests/dit/test_model.py::test_tt_asymm_dit_joint_inference -k "L1"
pytest models/experimental/mochi/tests/dit/test_model.py::test_tt_asymm_dit_joint_model_fn -k "L1"

pytest models/experimental/mochi/tests/dit/test_pipeline.py -k "L1"

# VAE tests
# pytest models/experimental/mochi/tests/vae/test_context_parallel_conv3d.py
# pytest models/experimental/mochi/tests/vae/test_conv1x1.py
# pytest models/experimental/mochi/tests/vae/test_depth_to_spacetime.py
# pytest models/experimental/mochi/tests/vae/test_silu.py
# pytest models/experimental/mochi/tests/vae/test_resblock.py
# pytest models/experimental/mochi/tests/vae/test_upsample.py
# pytest models/experimental/mochi/tests/vae/test_decoder.py
