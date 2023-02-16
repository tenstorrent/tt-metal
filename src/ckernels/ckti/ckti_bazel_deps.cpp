// CKTI - C Kernel Test Image
// This is an even SMALLER utility that cats the sources of all ckernel files needed by a test
// and feeds it as input to ckti actions 
//
// This is required to maintain finegrained dependencies for each versim test image; 
// If we don't use this, we must depend on ALL ckernel source files, such that if any ckernel changes, all versim tests 
// will be invalidated in the remote cache

#include <string>
#include <vector>
#include <sstream>
#include <set>
#include <tclap/CmdLine.h>
#include "command_assembler/archive.h"
#include "command_assembler/coreimage.h"
#include "command_assembler/kernels.h"
#include "command_assembler/systemimage.h"
#include "yaml-cpp/yaml.h"

using CommandAssembler::xy_pair;

static const std::string TTX_MANIFEST_FILENAME = "test.yaml";

template <typename F>
void statusf(F printer)
{
    std::cout << "= CKTI: ";
    printer(std::cout);
    std::cout << "\n";
    std::cout.flush();
}

void status(const std::string &msg)
{
    return statusf([&](auto &&s) { s << msg; });
}

template <typename F>
[[noreturn]] void errorf(F printer)
{
    std::stringstream s;
    s << "*\n* CKTI ERROR: ";
    printer(s);
    s << "\n*\n";
    std::cerr << s.str();
    std::cerr.flush();
    throw std::runtime_error(s.str());
}

[[noreturn]] void error(const std::string &msg)
{
    errorf([&](auto &&s) { s << msg; });
}

bool run_command(const std::string &cmd)
{
    status("running: `" + cmd + '`');
    int ret = system(cmd.c_str());
    return (ret == 0);
}

int main(int argc, char *argv[])
{
    TCLAP::CmdLine cmd("C Kernel Test Image bazel dependency generator", '=');
    TCLAP::ValueArg<std::string>  test_name("", "test", "Test name", true, "", "string", cmd);
    TCLAP::ValueArg<std::string>  test_dir("", "dir", "Test output directory", true, "", "directory", cmd);
    TCLAP::ValueArg<std::string>  ckernel_src_path("", "ckp", "CKernel Path", true, "", "directory", cmd);
    TCLAP::ValueArg<std::string> ckernel_out("","out", "CKernel cat path", false, "", "directory", cmd);
    cmd.parse(argc, argv);
    
    std::string testname = test_name.getValue();
    std::string testdir = test_dir.getValue();
    std::string ckernel_path = ckernel_src_path.getValue();
    std::string out_path =  ckernel_out.getValue().empty() ? testdir : ckernel_out.getValue();

    std::string filename = testdir + "/" + testname + ".ttx";

    CommandAssembler::ReadArchive arch(filename);
    CommandAssembler::SystemImage system_image = CommandAssembler::SystemImage::load(arch);
    const std::string yaml = "test.yaml";
    if (!arch.has_archive_entry(yaml)) {
        error("No " + yaml + " test definition found in the archive.");
        return 1;
    }

    YAML::Node ttx_manifest_mutable;
    try
    {
        ttx_manifest_mutable = YAML::Load(arch.read_archive_entry(TTX_MANIFEST_FILENAME));
    }
    catch (const std::exception &e)
    {
        errorf([&](auto &&s) { s << "Unable to load ttx manifest: " << e.what(); });
    }
    const YAML::Node &ttx_manifest = ttx_manifest_mutable;

    status("Parsing Manifest");
    const YAML::Node ckernel_node = ttx_manifest["ckernel"];
    if (not ckernel_node.IsDefined() || ckernel_node.IsNull())
    {
        error("Failed to find any ckernel info in ttx. Check " + TTX_MANIFEST_FILENAME + " in ttx.");
    }

    std::set<std::string> unique_ckernels;
    for (const auto &core_and_info : ckernel_node["by_core"])
    {
        for (const auto &thread_and_info : core_and_info.second["by_thread"])
        {
            for (const auto &ckernel_name_and_info : thread_and_info.second["by_ckernel"])
            {
                unique_ckernels.insert(ckernel_name_and_info.first.as<std::string>());
            }
        }
    }

    std::string catcmd = "cat ";
    for(auto& ckernel : unique_ckernels) {
        catcmd += ckernel_path + "/c" + ckernel + ".o/*.o ";
    }
    catcmd += ("> " + out_path + "/ckernels.out");
    if (not run_command(catcmd))
    {
        error("final cat failed!");
    }
}

