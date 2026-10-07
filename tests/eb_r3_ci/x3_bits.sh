#!/usr/bin/env bash
# Round 3 eltwise binary: whole modules of the extra callers bit for bit (optin_x3.txt); the llama rotary module's decode cases.
cd /work
O=tests/eb_r3_ci/optin_x3.txt
bash tests/eb_r3_ci/bits_ab.sh $O tests/ttnn/nightly/unit_tests/operations/experimental/indexer_score/test_indexer_score.py \
  models/demos/blackhole/qwen36/tests/test_fused_recurrent_gdn.py tests/eb_r3_ci/test_eb_gdn.py tests/eb_r3_ci/test_eb_x3.py
bash tests/eb_r3_ci/bits_ab.sh $O -p eb_unskip_plugin tests/tt_eager/python_api_testing/unit_testing/misc/test_rotary_embedding_llama_fused_qk.py \
  tests/ttnn/nightly/unit_tests/operations/moreh/test_moreh_layer_norm.py
bash tests/eb_r3_ci/bits_ab.sh $O -p eb_unskip_plugin tests/ttnn/nightly/unit_tests/operations/experimental/test_rotary_embedding_llama.py -k decode
