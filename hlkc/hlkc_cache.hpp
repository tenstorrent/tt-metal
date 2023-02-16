#pragma once

#include <array>
#include <assert.h>
#include <iostream>
#include <memory.h>
#include <stdio.h>
#include <stdlib.h>

#include <filesystem>
#include "meow_hash/meow_hash_x64_aesni.h"

#include "hlkc.hpp"

// Use only the below structures from hlkc_cache, but the actual caches
// for HLKC are not used

namespace fs = std::filesystem;
using std::string;

namespace hlkc {
namespace cache {

typedef struct meow_u128_hash {
    meow_u128 data;
    bool valid;

    meow_u128_hash() : valid(false) {} // data can be left as garbage

} tt_hash_128;
} // end namespace cache
} // end namespace hlkc

typedef struct _CompilationContext {

    // Input Files
    fs::path hlkc_path;
    fs::path hlk_file_name;
    fs::path llk_args_file_name;

    // Target
    LLKTarget llk_target;

    // Output Directory
    fs::path output_dir;
    fs::path output_file_name;

    // Remaining unparsed ROSE arguments
    typedef Rose_STL_Container<SgNode*> SgNodeContainer;
    Rose_STL_Container<string> arg_vec;

    // Compilation flags
    string device_name;
    bool enable_cache;
    bool perf_dump_en;
    bool untilize_output;
    bool pack_microblocks;
    bool fp32_dest_acc_en;
    uint32_t relu_config;
    bool pack_l1_acc_en;

    // Hash
    hlkc::cache::tt_hash_128 hash;

} CompilationContext;
