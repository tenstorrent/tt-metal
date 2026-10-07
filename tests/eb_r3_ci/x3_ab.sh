#!/usr/bin/env bash
# Round 3 eltwise binary: device time A/B of the extra callers (optin_x3.txt), one selection at a time.
cd /work
O=tests/eb_r3_ci/optin_x3.txt
I=tests/ttnn/nightly/unit_tests/operations/experimental/indexer_score/test_indexer_score.py
for sel in "$I -k accuracy" "$I -k shapes" \
  "models/demos/blackhole/qwen36/tests/test_fused_recurrent_gdn.py -k fused" \
  "tests/eb_r3_ci/test_eb_gdn.py" \
  "-p eb_unskip_plugin tests/ttnn/nightly/unit_tests/operations/experimental/test_rotary_embedding_llama.py -k decode" \
  "-p eb_unskip_plugin tests/tt_eager/python_api_testing/unit_testing/misc/test_rotary_embedding_llama_fused_qk.py" \
  "tests/eb_r3_ci/test_eb_x3.py" \
  "-p eb_unskip_plugin tests/ttnn/nightly/unit_tests/operations/moreh/test_moreh_layer_norm.py -k large_algorithm"; do
  echo "##### selection: $sel"
  bash tests/eb_r3_ci/ab_set.sh $O $sel
done
