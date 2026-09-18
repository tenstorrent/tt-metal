# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""CPU contract: compile the actual factory partition/selection/runtime-arg code.

No TTNN import or hardware. --noconftest avoids opening the device fixtures.
The small C++ stubs model only core coordinates and runtime-argument storage;
partitioning and filtering are extracted from the production factory itself.
"""

import shutil
import subprocess
from pathlib import Path

ROOT = next(p for p in Path(__file__).resolve().parents if (p / "ttnn/api/ttnn/device_operation.hpp").exists())
OP = ROOT / "ttnn/cpp/ttnn/operations/experimental/transformer/rotary_embedding_llama"


def test_active_targets_match_runtime_args_and_cover_every_sequence_tile(tmp_path):
    factory = (OP / "device/rotary_embedding_llama_multi_core_program_factory.cpp").read_text()
    partition = factory.split("    const uint32_t num_cores =", 1)[1].split(
        "    const uint32_t num_sin_cos_rows_per_core", 1
    )[0]
    partition = "const uint32_t num_cores =" + partition
    assignments = factory.split("    const auto& cores = grid_to_cores", 1)[1].split(
        "    // Assemble spec + run-args.", 1
    )[0]
    assignments = "const auto& cores = grid_to_cores" + assignments
    assert ".target_nodes = target_nodes" in factory
    source = (
        r"""
#include <algorithm>
#include <cassert>
#include <cstdint>
#include <map>
#include <set>
#include <string>
#include <tuple>
#include <variant>
#include <vector>
#define TT_FATAL(condition, ...) assert(condition)
using NodeCoord = uint32_t;
struct CoreRange {
    uint32_t lo, hi;
    CoreRange(uint32_t a, uint32_t b) : lo(a), hi(b) {}
    auto operator<=>(const CoreRange&) const = default;
};
struct CoreRangeSet { std::set<CoreRange> ranges; CoreRangeSet(std::set<CoreRange> r): ranges(r) {} };
using Nodes = std::variant<CoreRange, CoreRangeSet>;
using Values = std::map<uint32_t, std::vector<std::pair<std::string, uint32_t>>>;
struct KernelRunArgs { int kernel; Values runtime_arg_values; };
constexpr int READER=0, WRITER=1, COMPUTE=2;
void AddRuntimeArgsForNode(Values& values, uint32_t node, std::vector<std::pair<std::string,uint32_t>> args) {
    assert(values.emplace(node, args).second);
}
std::vector<uint32_t> grid_to_cores(uint32_t n, uint32_t, uint32_t, bool) {
    std::vector<uint32_t> result; for(uint32_t i=0;i<n;++i) result.push_back(i); return result;
}
Values run(uint32_t seq_len_t, bool active) {
    const uint32_t batch=1, num_cores_x=12, num_cores_y=10;
    const bool row_major=true;
    const CoreRange all_cores(0,119);
    struct {bool active_cores_only;} operation_attributes{active};
"""
        + partition
        + assignments
        + r"""
    std::set<uint32_t> targets;
    if(active) {
        for(const auto& range: std::get<CoreRangeSet>(target_nodes).ranges)
            for(uint32_t i=range.lo;i<=range.hi;++i) targets.insert(i);
    } else for(uint32_t i=0;i<120;++i) targets.insert(i);
    assert(reader_run.runtime_arg_values == writer_run.runtime_arg_values);
    assert(reader_run.runtime_arg_values == compute_run.runtime_arg_values);
    std::vector<int> writes(seq_len_t,0);
    for(const auto& [node,args]: reader_run.runtime_arg_values) {
        assert(targets.erase(node)==1);
        assert(args.size()==4);
        if(active) assert(args[0].second<args[1].second && args[2].second<args[3].second);
        for(uint32_t i=args[2].second;i<args[3].second;++i) ++writes.at(i);
    }
    assert(targets.empty());
    for(int count:writes) assert(count==1);
    return reader_run.runtime_arg_values;
}
int main() {
    for(uint32_t rows: {1,2,3,37,38,119,120,121,152,239,240,241,1024}) {
        auto legacy=run(rows,false), active=run(rows,true);
        for(const auto& [node,args]: active) assert(legacy.at(node)==args);
        for(const auto& [node,args]: legacy)
            assert(active.contains(node)==(args[0].second<args[1].second && args[2].second<args[3].second));
    }
    assert(run(1,true).size()==1);
    assert(run(38,true).size()==38);
    assert(run(152,true).size()==76);
}
"""
    )
    cpp, executable = tmp_path / "factory.cpp", tmp_path / "factory"
    cpp.write_text(source)
    compiler = shutil.which("c++")
    assert compiler, "a host C++20 compiler is required for this CPU source-contract test"
    subprocess.run([compiler, "-std=c++20", "-O0", str(cpp), "-o", str(executable)], check=True, timeout=60)
    subprocess.run([str(executable)], check=True, timeout=10)


def test_explicit_flag_participates_in_default_program_cache_identity():
    params = (OP / "device/rotary_embedding_llama_device_operation_types.hpp").read_text()
    dispatch = (OP / "device/rotary_embedding_llama_device_operation.cpp").read_text()
    header = (OP / "device/rotary_embedding_llama_device_operation.hpp").read_text()
    assert "bool active_cores_only = false;" in params
    assert ".active_cores_only = active_cores_only" in dispatch
    assert "compute_program_hash" not in header  # Uses framework aggregate-attribute hashing.
    framework = (ROOT / "ttnn/api/ttnn/device_operation.hpp").read_text()
    assert "ttsl::hash::type_hash<device_operation_t>, operation_attributes, tensor_args" in framework
    factory = (OP / "device/rotary_embedding_llama_multi_core_program_factory.cpp").read_text()
    assert "getenv" not in factory  # No hidden env-sensitive factory/cache identity.
