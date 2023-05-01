#include "common/utils.hpp"

namespace tt
{
namespace utils
{
    bool run_command(const string &cmd, const string &log_file, const bool verbose)
    {
        int ret;
        if (getenv("TT_BACKEND_DUMP_RUN_CMD") or verbose) {
            cout << "===== RUNNING SYSTEM COMMAND:" << std::endl;
            cout << cmd << std::endl << std::endl;
            ret = system(cmd.c_str());
        } else {
            string redirected_cmd = cmd + " >> " + log_file + " 2>&1";
            ret = system(redirected_cmd.c_str());
        }
        return (ret == 0);
    }

    void create_file(string file_path_str) {
        fs::path file_path(file_path_str);
        fs::create_directories(file_path.parent_path());

        std::ofstream ofs(file_path);
        ofs.close();
    }

    std::string get_root_dir() {
        constexpr std::string_view ROOT_DIR_ENV_VAR = "TT_METAL_HOME";
        std::string root_dir;

        if(const char* root_dir_ptr = std::getenv(ROOT_DIR_ENV_VAR.data())) {
            root_dir = root_dir_ptr;
        } else {
            TT_THROW("Env var " + std::string(ROOT_DIR_ENV_VAR) + " is not set.");
        }
        return root_dir;
    }
    std::string get_env_arch_name() {
        constexpr std::string_view ARCH_NAME_ENV_VAR = "ARCH_NAME";
        std::string arch_name;

        if(const char* arch_name_ptr = std::getenv(ARCH_NAME_ENV_VAR.data())) {
            arch_name = arch_name_ptr;
        } else {
            TT_THROW("Env var " + std::string(ARCH_NAME_ENV_VAR) + " is not set.");
        }
        return arch_name;
    }
}
}
