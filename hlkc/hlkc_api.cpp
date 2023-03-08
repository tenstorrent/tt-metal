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
    string buda_home;
    if (!getenv("TT_METAL_HOME")) {
        fs::path cwd = fs::current_path();
        buda_home = cwd.string();
        hlkc_path = buda_home + "/build/bin/hlkc";
    } else {
        buda_home = string(getenv("TT_METAL_HOME"));
        hlkc_path = buda_home + "/build/bin/hlkc";
    }
    cout << "    HLK file name: " << input_hlk_file_path << " HLKC_PATH " << hlkc_path << endl;
    string root_dir = "";       // drago - this should be ok since input_hlk_file_path and out_dir_path will be passed with TT_METAL_HOME included
    string out_file_name_base = "chlkc";

    string unpack_base   = out_dir_path + "/" + out_file_name_base + "_unpack";
    string math_base     = out_dir_path + "/" + out_file_name_base   + "_math";
    string pack_base     = out_dir_path + "/" + out_file_name_base  + "_pack";

    string unpack_cpp         = unpack_base + ".cpp";
    string unpack_llk_args_h  = unpack_base + "_llk_args.h";
    string math_cpp           = math_base + ".cpp";
    string math_llk_args_h    = math_base + "_llk_args.h";
    string pack_cpp           = pack_base + ".cpp";
    string pack_llk_args_h    = pack_base + "_llk_args.h";

    string unpack_log   = unpack_base + ".log";
    string math_log     = math_base + ".log";
    string pack_log     = pack_base + ".log";

    string perf_dump_flag = "-perf_dump:" + to_string(dump_perf_events);
    string untilize_output_flag = "-untilize_output:" + to_string(untilize_output);
    string pack_microblocks_flag = "-pack_microblocks:" + to_string(pack_microblocks);
    string fp32_dest_acc_en_flag = "-fp32_dest_acc_en:" + to_string(fp32_dest_acc_en);
    string device_name_flag = "-device_name:" + device_name;

    int status;
    string cache_setting;
    if (enable_cache) {
        cache_setting = "-cache:on";
    } else {
        cache_setting = "-cache:off";
    }

    string input_hlk_with_defines = input_hlk_file_path;
    {
        // Here we generate an auxiliary header with defines added via add_define() call
        // this header is then included from the kernel
        // We also append the include path to generated dir to hlkc cmldline.
        std::ofstream gen_defines_file;
        string generated_defines_fname = out_dir_path + "/hlk_defines_generated.h";
        gen_defines_file.open(generated_defines_fname, std::ios_base::out);

        for (auto it = defines.begin(); it != defines.end(); ++it) {
            gen_defines_file << "#define " << it->first << " " << it->second << endl;
        }

        // this string will be returned from the function to be reused in subsequent calls
        input_hlk_with_defines += " -I" + out_dir_path + " ";
        hlkc_path += " " + input_hlk_with_defines;
    }

    // For any common code that we need from kernels dir itself
    string kernels_include = "-I" + buda_home + "/kernels/ -I" + buda_home + "/kernels/hostdevcommon/ ";

    string compile_unpack = hlkc_path + " -hlkc:llk_target unpack -rose:o " + unpack_cpp + " -hlkc:llk_args " + unpack_llk_args_h + " " + cache_setting + " " + perf_dump_flag + " " + fp32_dest_acc_en_flag + " " + device_name_flag + " " + kernels_include + " > " + unpack_log + " 2>&1";
    string compile_math = hlkc_path + " -hlkc:llk_target math -rose:o " + math_cpp + " -hlkc:llk_args " + math_llk_args_h + " " + cache_setting + " " + perf_dump_flag + " " + fp32_dest_acc_en_flag + " " + device_name_flag+ " " + kernels_include + " > " + math_log + " 2>&1";
    string compile_pack = hlkc_path + " -hlkc:llk_target pack -rose:o " + pack_cpp + " -hlkc:llk_args " + pack_llk_args_h + " " + cache_setting + " " + perf_dump_flag + " " + untilize_output_flag + " " + pack_microblocks_flag +  " " + fp32_dest_acc_en_flag + " " + device_name_flag + " " + kernels_include + " > " + pack_log + " 2>&1";

    vector<string> cmds = {compile_unpack, compile_math, compile_pack};
    vector<thread*> threads;
    function<void(string, int)> sys_lambda = [] (string cmd, int phase) {
        cout << "    " << cmd << endl;
        int status = system(cmd.c_str());
        // could include hlkc.hpp here for LLKTarget enum but it seems like these changes
        // are localized so a phase index is self-explanatory without pulling lots of deps.
        switch (phase) {
            case  0: assert(status==0 && "Error: HLKC unpack compile failed."); break;
            case  1: assert(status==0 && "Error: HLKC math compile failed.");   break;
            default: assert(status==0 && "Error: HLKC pack compile failed.");   break;
        }
    };
    if (parallel) {
        for (int icmd = 0; icmd < cmds.size(); icmd++)
            threads.push_back(new thread(sys_lambda, cmds[icmd], icmd));
        for (auto th: threads) {
            th->join();
            delete th;
        }
    } else {
        sys_lambda(compile_unpack, 0);
        sys_lambda(compile_math, 1);
        sys_lambda(compile_pack, 2);
    }

    return input_hlk_with_defines;
}

void compile_generate_struct_init_header(
    std::string input_hlk_file_path, std::string out_dir_path, std::string out_file_name_base, bool enable_cache = true) {
    int status;
    string cache_setting;
    if (enable_cache) {
        cache_setting = "-cache:on";
    } else {
        cache_setting = "-cache:off";
    }
    string hlkc_path;
    string buda_home;
    if (!getenv("TT_METAL_HOME")) {
        fs::path cwd = fs::current_path();
        buda_home = cwd.string();
        hlkc_path = cwd.string() + "/build/bin/hlkc";
    } else {
        buda_home = string(getenv("TT_METAL_HOME"));
        hlkc_path = buda_home + "/build/bin/hlkc";
    }

    string generate_struct_init_cpp = out_dir_path + "/" + out_file_name_base + "_gen.cpp";
    string generate_struct_init_log = out_dir_path + "/" + out_file_name_base + "_gen.log";
    string generate_struct_init_o = out_dir_path + "/" + out_file_name_base   + "_gen.o";
    string generate_struct_init_so = out_dir_path + "/" + out_file_name_base  + "_gen.so";
    string generate_struct_init_soname = out_file_name_base + ".so";

    // For any common code that we need from kernels dir itself
    string kernels_include = "-I" + buda_home + "/kernels/ -I" + buda_home + "/kernels/hostdevcommon/ ";

    string generate_generator_cmd = hlkc_path + " " + input_hlk_file_path + " -hlkc:llk_target struct_init_gen -rose:o " + generate_struct_init_cpp + " " + cache_setting + " " + kernels_include + " > " + generate_struct_init_log + " 2>&1";
    cout << "    " << generate_generator_cmd << endl;
    status = system(generate_generator_cmd.c_str());
    assert(status==0 && "Error: HLKC generate struct init generator failed.");
    //status = system("./hlkc ../hlks/eltwise_unary_datacopy_full_dst_mode.cpp -hlkc:llk_target struct_init_gen -rose:o out/llks/eltwise_unary_struct_init_gen.cpp");

    string compile_obj_generator_cmd = "g++ -fPIC -g -c -Wall " + generate_struct_init_cpp + " -o " + generate_struct_init_o;
    cout << "    " << compile_obj_generator_cmd << endl;
    status = system(compile_obj_generator_cmd.c_str());
    assert(status==0 && "Error: struct init generator g++ compile object failed.");
    //status = system("g++ -fPIC -g -c -Wall out/llks/eltwise_unary_struct_init_gen.cpp -o out/llks/eltwise_unary_struct_init_gen.o");

    string create_shared_obj_generator_cmd =  "g++ -shared -Wl,-soname," + generate_struct_init_soname + " -o " + generate_struct_init_so + " " + generate_struct_init_o + " -lc";
    cout << "    " << create_shared_obj_generator_cmd << endl;
    status = system(create_shared_obj_generator_cmd.c_str());
    assert(status==0 && "Error: struct init generator g++ compile shared object failed.");
    //status = system("g++ -shared -Wl,-soname,eltwise_unary_struct_init_gen.so.1 -o out/llks/eltwise_unary_struct_init_gen.so.1.0 out/llks/eltwise_unary_struct_init_gen.o -lc");
}

int run_generate_struct_init_header(
    std::string out_dir_path, std::string out_file_name_base, const void *args
) {
    string generator_lib_file_name = out_dir_path + "/" + out_file_name_base + "_gen.so";
    string out_header_file_name = out_dir_path + "/" + out_file_name_base + ".h";

    void *lib_handle;
    void (*func)(const void*, const char*);
    char *error;

    lib_handle = dlopen(generator_lib_file_name.c_str(), RTLD_LAZY);

    if (!lib_handle) {
       fprintf(stderr, "%s\n", dlerror());
       exit(1);
    }

    func = (void (*)(const void *, const char*)) dlsym(lib_handle, "generate_hlk_args_struct_init");

    if ((error = dlerror()) != NULL) {
        fprintf(stderr, "%s\n", error);
        exit(1);
    }

    (*func)(args, out_header_file_name.c_str());

    dlclose(lib_handle);

    return 0;
}
