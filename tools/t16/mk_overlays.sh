#!/bin/bash
# tools/t16/mk_overlays.sh : (re)generate models/demos/blackhole/deepseek_v41_flash/tt/t16_kernels/{state2,all} from their sources.
#   sdpa/   = the T16 sparse_sdpa K-row-cache kernels (hand-maintained here)
#   state2/ = T15 stateful-NoC-read weight_runs.hpp for the unified routed-expert op (+ the unchanged image reader/writer .cpp, which a TT_METAL_KERNEL_PATH overlay must carry because the headers
#             are found relative to the including file)
#   all/    = both
# Usage on a node: TT_METAL_KERNEL_PATH=<model dir>/tt/t16_kernels/<name> (ONLY effective when the process working directory does not contain ttnn/cpp/...: the server's cwd is
# /home/container_app_user/app/src; kernel sources are resolved cwd, TT_METAL_KERNEL_PATH, system dir, TT_METAL_HOME). The JIT cache key does not track included headers: every overlay .cpp carries a
# '// T16-HDR-SHA' comment (sha of the overlay's headers) so a changed header changes the kernel hash.
set -e
R=$(cd $(dirname $0)/../..; pwd); T=$R/models/demos/blackhole/deepseek_v41_flash/tt/t16_kernels; S=$R/tools/t16/src
U=ttnn/cpp/ttnn/operations/experimental/deepseek_prefill/unified_routed_expert_ffn/device/kernels
SD=ttnn/cpp/ttnn/operations/transformer/sdpa/device/kernels/dataflow
mkdir -p $T/state2/$U/dataflow
cp $S/weight_runs_state2.hpp $T/state2/$U/weight_runs.hpp
cp $S/adaptive_chunk.hpp $T/state2/$U/adaptive_chunk.hpp
cp $S/unified_routed_expert_ffn_reader.cpp $S/unified_routed_expert_ffn_writer.cpp $T/state2/$U/dataflow/
stamp() { # stamp <overlay dir>: append the header sha comment to every .cpp
  d=$1; sha=$(cat $(find $d -name '*.hpp' | sort) | sha256sum | cut -c1-16)
  for f in $(find $d -name '*.cpp'); do sed -i '/^\/\/ T16-HDR-SHA/d' $f; echo "// T16-HDR-SHA $sha" >> $f; done
}
stamp $T/state2; stamp $T/sdpa
rm -rf $T/all; mkdir -p $T/all; cp -a $T/sdpa/. $T/all/; cp -a $T/state2/. $T/all/
find $T -type f | sort | xargs sha256sum
