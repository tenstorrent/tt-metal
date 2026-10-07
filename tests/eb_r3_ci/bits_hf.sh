#!/usr/bin/env bash
# Round 3 eltwise binary (#58723 review): the HF rotary modules bit for bit with the standard-multiply switch on top of the broadcast one.
cd /work
bash tests/eb_r3_ci/bits_ab.sh tests/eb_r3_ci/optin_rep_hf.txt -p eb_seed_plugin tests/tt_eager/python_api_testing/unit_testing/misc/test_rotary_embedding_hf.py
