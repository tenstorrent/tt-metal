#!/bin/bash
# t449 (CPU only, runs on blx01): incremental Release rebuild of /var/tmp/fasth3/t409/b after the vsa_sdpa
# exempt-cap fix (host C++), then check the VSA ops still bind. Idle I/O class, nice, 24 jobs.
set -eo pipefail
F=/var/tmp/fasth3; A=$F/t48; T=$F/t409; B=$T/b
export HOME=$F/home XDG_CACHE_HOME=$F/home/.cache TMPDIR=$F/tmp CPM_SOURCE_CACHE=$F/.cpmcache CMAKE_BUILD_PARALLEL_LEVEL=24
set -x
use=$(df --output=pcent / | tail -1 | tr -dc 0-9); [ "$use" -le 85 ]
grep -q 'more exempt block ids' $B/ttnn/cpp/ttnn/operations/transformer/sdpa/device/vsa_sdpa_device_operation.cpp
cd $B
ionice -c3 nice -n 10 ./build_metal.sh --build-type Release
source $A/python_env/bin/activate
cd /tmp && TT_METAL_HOME=$B PYTHONPATH=$B:$B/ttnn python -c "
import ttnn, os
assert os.path.dirname(ttnn.__file__) == '$B/ttnn/ttnn', ttnn.__file__
for n in ('vsa_sdpa', 'vsa_ring_sdpa'):
    assert hasattr(ttnn.transformer, n), n
print('VSA ops bound')"
echo BUILD449_DONE
