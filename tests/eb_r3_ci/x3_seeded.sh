#!/usr/bin/env bash
# Round 3 eltwise binary: the x3 selections whose bits differed, every test seeded; main against main, then main against the opt-ins.
cd /work
for O in tests/eb_r3_ci/optin_none.txt tests/eb_r3_ci/optin_x3.txt; do
  echo "##### $O"
  bash tests/eb_r3_ci/bits_ab.sh $O -p eb_seed_plugin models/demos/blackhole/qwen36/tests/test_fused_recurrent_gdn.py tests/eb_r3_ci/test_eb_gdn.py
  bash tests/eb_r3_ci/bits_ab.sh $O -p eb_seed_plugin -p eb_unskip_plugin tests/ttnn/nightly/unit_tests/operations/experimental/test_rotary_embedding_llama.py -k decode
done
