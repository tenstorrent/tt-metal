#include "common/utils.hpp"

namespace tt
{
namespace utils
{
    bool run_command(const string &cmd, const string &log_file, const bool verbose)
    {
        int ret;
        if (getenv("TT_BACKEND_DUMP_RUN_CMD") or verbose) {
            cout << "running: `" + cmd + '`';
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
        constexpr std::string_view ROOT_DIR_ENV_VAR = "BUDA_HOME";
        std::string root_dir;

        if(const char* root_dir_ptr = std::getenv(ROOT_DIR_ENV_VAR.data())) {
            root_dir = root_dir_ptr;
        } else {
            TT_THROW("Env var " + std::string(ROOT_DIR_ENV_VAR) + " is not set.");
        }
        return root_dir;
    }
}
}
