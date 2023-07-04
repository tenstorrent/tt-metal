#include "common/utils.hpp"
#include <mutex>
#include "tt_metal/third_party/tracy/public/tracy/Tracy.hpp"

namespace tt
{
namespace utils
{
    bool run_command(const string &cmd, const string &log_file, const bool verbose)
    {
        ZoneScoped;
        ZoneText( cmd.c_str(), cmd.length());
        int ret;
        static std::mutex io_mutex;
        if (getenv("TT_BACKEND_DUMP_RUN_CMD") or verbose) {
            {
                std::lock_guard<std::mutex> lk(io_mutex);
                cout << "===== RUNNING SYSTEM COMMAND:" << std::endl;
                cout << cmd << std::endl << std::endl;
            }
            ret = system(cmd.c_str());
            // {
            //     std::lock_guard<std::mutex> lk(io_mutex);
            //     cout << "===== DONE SYSTEM COMMAND: " << cmd << std::endl;
            // }

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

    const std::string& get_reports_dir() {
        static const std::string outpath = get_root_dir() + "/.reports/";
        return outpath;
    }
}
}
