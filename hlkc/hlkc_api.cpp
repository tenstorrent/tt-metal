#include <cstdint>
#include <dlfcn.h>
#include <cstdio>
#include <cstdlib>
#include <string>
#include <iostream>
#include <cassert>
#include <thread>
#include <functional>
#include <filesystem>
#include <fstream>
#include <map>
#include <vector>
#include <sstream>

#include "hlkc_api.h"

using namespace std;
using std::string;
using std::cout;
using std::vector;
namespace fs = std::filesystem;

void gen_trisc_cpp(const string& src_name, const string& dst_name, vector<string>& prolog)
{
    //std::ifstream in(src_name);
    //std::stringstream buffer;
    //buffer << in.rdbuf();

    std::ofstream out(dst_name);
    for (auto s: prolog)
        out << s;
    out << "#include \"" << src_name << "\"\n";
    //out << buffer.str();
}

// returns the name of temporary hlk .cpp with generated #defines in the beginning
std::string compile_hlk(
    string input_hlk_file_path,
    string out_dir_path,
    string device_name,
    const hlk_defines_map_t& defines,
    bool dump_perf_events,
    bool untilize_output,
    bool enable_cache,
    bool pack_microblocks,
    bool fp32_dest_acc_en,
    bool parallel
)
{
    string hlkc_path;
    string tt_metal_home;
    if (!getenv("TT_METAL_HOME")) {
        fs::path cwd = fs::current_path();
        tt_metal_home = cwd.string();
        hlkc_path = tt_metal_home + "/build/bin/hlkc";
    } else {
        tt_metal_home = string(getenv("TT_METAL_HOME"));
        hlkc_path = tt_metal_home + "/build/bin/hlkc";
    }

    auto old_cwd = fs::current_path();
    fs::current_path(tt_metal_home); // make build work for any cwd

    string out_file_name_base = "chlkc";
    string unpack_base        = out_dir_path + "/" + out_file_name_base + "_unpack";
    string math_base          = out_dir_path + "/" + out_file_name_base   + "_math";
    string pack_base          = out_dir_path + "/" + out_file_name_base  + "_pack";
    string unpack_cpp         = unpack_base + ".cpp";
    string unpack_llk_args_h  = unpack_base + "_llk_args.h";
    string math_cpp           = math_base + ".cpp";
    string math_llk_args_h    = math_base + "_llk_args.h";
    string pack_cpp           = pack_base + ".cpp";
    string pack_llk_args_h    = pack_base + "_llk_args.h";

    vector<string> unpack_prolog;
    unpack_prolog.push_back("#define TRISC_UNPACK\n");
    unpack_prolog.push_back("#include \"defines_generated.h\"\n");
    vector<string> math_prolog;
    math_prolog.push_back("#define TRISC_MATH\n");
    math_prolog.push_back("#include \"defines_generated.h\"\n");
    vector<string> pack_prolog;
    pack_prolog.push_back("#define TRISC_PACK\n");
    pack_prolog.push_back("#include \"defines_generated.h\"\n");

    std::thread t0( [&]() { gen_trisc_cpp(input_hlk_file_path, unpack_cpp, unpack_prolog); } );
    std::thread t1( [&]() { gen_trisc_cpp(input_hlk_file_path, math_cpp, math_prolog); } );
    std::thread t2( [&]() { gen_trisc_cpp(input_hlk_file_path, pack_cpp, pack_prolog); } );
    t0.join(); t1.join(); t2.join();

    string input_hlk_with_defines = input_hlk_file_path;
    {
        // Here we generate an auxiliary header with defines added via add_define() call
        // this header is then included from the kernel
        // We also append the include path to generated dir to hlkc cmldline.
        std::ofstream gen_defines_file;
        string generated_defines_fname = out_dir_path + "/defines_generated.h";
        gen_defines_file.open(generated_defines_fname, std::ios_base::out);

        for (auto it = defines.begin(); it != defines.end(); ++it) {
            gen_defines_file << "#define " << it->first << " " << it->second << endl;
        }

        // this string will be returned from the function to be reused in subsequent calls
        input_hlk_with_defines += " -I" + out_dir_path + " ";
        hlkc_path += " " + input_hlk_with_defines;
    }

    fs::current_path(old_cwd); // restore cwd

    return input_hlk_with_defines;
}
