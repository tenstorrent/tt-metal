"""Compile the production receiver with CPU dataflow stubs; no TTNN import/device access."""

import json
import pathlib
import subprocess
import tempfile

ROOT = pathlib.Path(__file__).resolve().parents[4]
OPS = ROOT / "ttnn/cpp/ttnn/operations"
source = (OPS / "ccl/kernel_common/worker_sync_utils.hpp").read_text()
start = source.index("void advance_start_page_idx(")
end = source.index("\nstruct ReduceScatterOpReceiver")
production = source[start:end]
preamble = r"""
#include <array>
#include <vector>
#include <cassert>
#include <cstdint>
#include <stdexcept>
#include "ttnn/operations/ccl/shared_with_host/linear_full_slice_schedule.hpp"
#define ASSERT(x) do { if (!(x)) throw std::runtime_error(#x); } while (false)
std::vector<uint32_t> args;
std::vector<std::array<uint32_t,2>> waits;
template<class T> T get_arg_val(uint32_t i) { return args.at(i); }
template<class T = void> struct Semaphore {
 uint32_t id;
 Semaphore(uint32_t id):id(id){}
 void wait_min(uint32_t value) { waits.push_back({id,value}); }
};
"""
main = r"""
int main() {
#ifdef LINEAR_FULL_SLICE_V1
 const uint32_t p = 4;
 const uint32_t sources[4][4] = {{0,1,2,3},{1,0,2,3},{2,1,3,0},{3,2,1,0}};
 const uint32_t dirs[4][4] = {{1,1,1,1},{1,0,1,1},{1,0,1,0},{1,0,0,0}};
 const uint32_t thresholds[4][4] = {{1,2,3,4},{1,1,2,3},{1,1,2,2},{1,1,2,3}};
#else
 const uint32_t p = 8;
#endif
 for (uint32_t rank = 0; rank < p; ++rank) {
  for (uint32_t block_k : {4u,8u}) {
   for (uint32_t m : {1u,32u}) {
    (void)m; // Logical M is independently host-validated; both occupy one physical tile.
    for (uint32_t operand : {0u,1u}) {
     const uint32_t stride = operand ? 40*272 : 40;
     const uint32_t step = operand ? block_k*272 : block_k;
     for (uint32_t base : {0u,34u,270u}) {
#ifdef LINEAR_FULL_SLICE_V1
      args={p,p,rank,40,stride,(p-1)*stride,1,11,12};
#else
      args={4,p,rank,40,stride,(p-1)*stride,1,11,12};
#endif
      waits.clear(); uint32_t cursor=0;
      MatmulOpReceiver receiver(!operand,cursor,p*40/block_k,block_k);
#ifdef LINEAR_FULL_SLICE_V1
      assert(cursor==9);
#else
      assert(cursor==(operand ? 7u : 9u)); // Preserve existing Ring ABI exactly.
#endif
      uint32_t address=base;
      std::array<bool,8> seen{};
      for (uint32_t block=0;block<p*40/block_k;++block) {
       receiver.update_current_block_start_tile_id(block,address,base);
       uint32_t ordinal=block/(40/block_k);
#ifdef LINEAR_FULL_SLICE_V1
       uint32_t expected=sources[rank][ordinal];
#else
       uint32_t expected= ordinal%2==0 ? (rank+ordinal/2)%p : (rank+p-(ordinal+1)/2)%p;
#endif
       assert(address==base+expected*stride+(block%(40/block_k))*step);
       if (block%(40/block_k)==0) {
        assert(!seen[expected]);seen[expected]=true;
        if (!operand) {
         assert(waits.size()==ordinal+1);
#ifdef LINEAR_FULL_SLICE_V1
         assert(waits.back()[0]==11+dirs[rank][ordinal]);
         assert(waits.back()[1]==thresholds[rank][ordinal]);
         assert(waits.back()[1] <= (dirs[rank][ordinal] ? p-rank : rank));
#else
         assert(waits.back()[0]==11+(ordinal%2==0));
         assert(waits.back()[1]==ordinal/2+1);
#endif
        }
       }
       address+=step;
      }
      for (uint32_t i=0;i<p;++i) assert(seen[i]);
      if (operand) assert(waits.empty());
     }
    }
   }
  }
 }
#ifdef LINEAR_FULL_SLICE_V1
 // The actual parser rejects invalid full-slice count, nondivisible block and empty slice.
 for (uint32_t negative=0;negative<3;++negative) {
  args={4,4,0,40,40,120,1,11,12};
  uint32_t k=4, blocks=40, cursor=0;
  if (negative==0) args[0]=8;
  if (negative==1) k=3;
  if (negative==2) args[3]=0;
  bool rejected=false;
  try { MatmulOpReceiver r(true,cursor,blocks,k); } catch (const std::runtime_error&) {rejected=true;}
  assert(rejected);
 }
#endif
}
"""
with tempfile.TemporaryDirectory(prefix="linear_agmm_cpu_") as tmp:
    tmp = pathlib.Path(tmp)
    cpp = tmp / "receiver.cpp"
    cpp.write_text(preamble + production + main)
    for mode in ("ring", "linear"):
        binary = tmp / mode
        command = [
            "g++",
            "-std=c++17",
            "-O2",
            "-Wall",
            "-Wextra",
            "-I",
            str(ROOT / "ttnn/cpp"),
            str(cpp),
            "-o",
            str(binary),
        ]
        if mode == "linear":
            command.insert(1, "-DLINEAR_FULL_SLICE_V1=1")
        subprocess.run(command, check=True)
        subprocess.run([str(binary)], check=True)
print(
    json.dumps(
        {
            "pass": True,
            "actual_receiver_compiled": True,
            "linear_cases": 96,
            "ring_cases": 192,
            "linear_negative_cases": 3,
            "device_validation": "pending",
        },
        indent=2,
    )
)
