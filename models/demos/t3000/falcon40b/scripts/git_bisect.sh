#!/bin/bash

# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

# Bisect metal to find the commit that makes CI fail
# Run by using git bisect:
#     mkdir ci
#     cd ci
#     bash /home/$USER/clone_metal.sh
#     cp <THIS_SCRIPT> .
#     cd tt-metal
#     git bisect start <earliest bad commit, e.g. main> <latest good commit>
#     git bisect run ./git_bisect.sh &>bisect_ci.log & disown -a

cd $TT_METAL_HOME

# git clean -fdx  # Beware: if you're using a machine lock you can't git clean unless you've checked this in your branch
git submodule update --recursive
REV=`git rev-parse HEAD`
echo "Testing revision: $REV"

echo "Building metal..."
source /home/$USER/setup-metal.sh
export TT_METAL_HOME=/home/${USER}/ci/tt-metal
/home/$USER/build-metal.sh &>../build-${REV}.log || exit 125   # an exit code of 125 asks "git bisect" to skip

echo "Resetting devices..."
source /opt/tt_metal_infra/provisioning/provisioning_env/bin/activate; tt-smi -r 0,1,2,3
deactivate

# Run the test
echo "Running test on $REV"
source python_env/bin/activate; pytest ${TT_METAL_HOME}/models/demos/t3000/falcon40b/tests/test_falcon_causallm.py::test_FalconCausalLM_inference[BFLOAT8_B-DRAM-falcon_40b-layers_1-prefill_seq32-8chips] &>../run-${REV}.log

if [ $? -eq 0 ]; then  # Test passed
    deactivate
    echo "$REV: 0 (test passed)"
    echo "$REV: 0 (test passed)" >> ../bisect.log
    exit 0
else  # Test failed
    deactivate
    echo "$REV: 1 (test failed)"
    echo "$REV: 1 (test failed)" >> ../bisect.log
    exit 1
fi
