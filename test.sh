#!/bin/bash
#SBATCH --nodes=17
#SBATCH --partition=debug
#SBATCH --job-name=test
#SBATCH --output=test_%j.out
#SBATCH --error=test_%j.err

# Set environmental variables
export TT_METAL_HOME="/data/ttuser/rsong/tt-metal"
export PYTHONPATH="/data/ttuser/rsong/tt-metal"
source python_env/bin/activate

tt-run --verbose ./build/test/tt_metal/tt_fabric/fabric_unit_tests --gtest_filter="MultiHost.TestClosetBoxTTSwitchControlPlaneInit"
