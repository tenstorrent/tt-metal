#!/usr/bin/env bash
# item 6: caller kernels, PR against main (each kernel's opt-in define removed); then softmax with both of its defines removed
bash tests/eb_r3_ci/dump_kedit.sh tests/eb_r3_ci/tog_callers.txt tests/eb_r3_ci/test_eb_dump_callers.py
EB_KEDIT_DIR=/tmp/ebkedit_sm bash tests/eb_r3_ci/dump_kedit.sh tests/eb_r3_ci/tog_softmax_both.txt tests/eb_r3_ci/test_eb_dump_callers.py -k test_softmax
