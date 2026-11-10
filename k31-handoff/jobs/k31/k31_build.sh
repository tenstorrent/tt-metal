#!/bin/bash
#SBATCH -N 1
#SBATCH -t 3:00:00
# k31_build.sh <tree> <commit>: offline build (no GitHub/PyPI from exabox). As slurm_job2.sh's build, with host Tracy off
# (--disable-profiler): the Tracy WASM viewer is an ExternalProject whose nested CMake fetches from GitHub.
set -euo pipefail
T=$1 COMMIT=$2
cd "$T"
[ "$(git rev-parse HEAD)" = "$(git rev-parse "$COMMIT^{commit}")" ] || { echo "tree is not at $COMMIT"; exit 5; }
export GIT_CONFIG_COUNT=1 GIT_CONFIG_KEY_0=url.file:///nonexistent-offline/.insteadOf GIT_CONFIG_VALUE_0=https://github.com/
export SETUPTOOLS_SCM_PRETEND_VERSION=0.75.0.dev0
rm -f /data/$USER/jobs/.built.$(basename "$T")
# --disable-profiler (ENABLE_TRACY=OFF) returns before the Tracy WASM viewer, whose nested CMake fetches its own
# packages from GitHub. Host Tracy is not used by these runs.
./build_metal.sh --build-tests --disable-profiler
echo "$COMMIT" > /data/$USER/jobs/.built.$(basename "$T")
"$T/python_env/bin/python" -c "import ttnn; print('ttnn ok', ttnn.__file__)"
echo "k31 build ok $COMMIT"
