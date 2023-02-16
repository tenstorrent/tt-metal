
#include <string>
#include <map>

using hlk_defines_map_t = std::map<std::string, std::string>;

// returns the name of temporary hlk .cpp with generated #defines in the beginning
std::string compile_hlk(std::string input_hlk_file_path,
                 std::string out_dir_path,
                 std::string device_name,
                 const hlk_defines_map_t& defines,
                 bool dump_perf_events,
                 bool untilize_output,
                 bool enable_cache = true,
                 bool pack_microblocks = false,
                 bool fp32_dest_acc_en = false,
                 bool parallel = false);
void compile_generate_struct_init_header(std::string input_hlk_file_path, std::string out_dir_path, std::string out_file_name_base, bool enable_cache);
int run_generate_struct_init_header(std::string out_dir_path, std::string out_file_name_base, const void *args);

