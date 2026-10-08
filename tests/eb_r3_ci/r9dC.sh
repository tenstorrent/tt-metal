#!/usr/bin/env bash
[[ -n ${HWLOCK_HELD:-} || -n ${GITHUB_ACTIONS:-} ]] || { echo "not under hwlock" >&2; exit 2; }
# Round 3 eltwise binary, sixth pass, merged head (sfpi 7.86.0): the opted-in kernels still taking the hand-off, defines removed against kept: ttnn.bcast (block-sharded H and HW), rotate_half, the 2D layernorm pre-all-gather, the HF rotary, indexer_score and softmax.
cd /work
T=tests/eb_r3_ci
export EB_RUN_LIMIT=1500
EB_KEDIT_DIR=/tmp/ebk_bc bash $T/dump_kedit.sh $T/tog_bcast_r9.txt $T/test_eb_dump_kedit.py -k "test_bcast or test_rotate_half"
EB_KEDIT_DIR=/tmp/ebk_misc bash $T/dump_kedit.sh $T/tog_callers_r9.txt $T/test_eb_dump_callers.py -k "test_rope_hf or test_indexer"
EB_KEDIT_DIR=/tmp/ebk_sm bash $T/dump_kedit.sh $T/tog_softmax_r9.txt $T/test_eb_dump_callers.py -k test_softmax
EB_KEDIT_DIR=/tmp/ebk_2d bash $T/dump_kedit.sh $T/tog_callers_r9.txt $T/test_eb_dump_callers.py -k "test_ln_pre_2d"
