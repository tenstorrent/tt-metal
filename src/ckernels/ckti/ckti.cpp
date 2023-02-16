// CKTI - C Kernel Test Image
// This is a small utility that links C kernels needed by a test,
// and adds them to the test image

#include <algorithm>
#include <mutex>
#include <sstream>
#include <string>
#include <tclap/CmdLine.h>
#include <vector>
#include <yaml-cpp/yaml.h>

#include "command_assembler/archive.h"
#include "command_assembler/binfile.h"
#include "command_assembler/coreimage.h"
#include "command_assembler/kernels.h"
#include "command_assembler/memory.h"
#include "command_assembler/systemimage.h"
#include "command_assembler/yaml_conversions.h"

using CommandAssembler::TensixCoreImage;
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

struct TriscImageAnalysis
{
    std::size_t max_allowed_size;
    std::size_t num_used_bytes;

    bool is_usable() const
    {
        return not is_oversize();
    }
    bool is_oversize() const
    {
        return num_used_bytes > max_allowed_size;
    }
};

struct TriscImageBuildResult
{
    TensixCoreImage image;
    std::string hex_file;
    std::string fwlog_file;
    TriscImageAnalysis analysis;
};

struct CoreThreadInfo
{
    std::vector<std::string> ckernel_names;
    bool operator==(const CoreThreadInfo &rhs) const
    {
        return ckernel_names == rhs.ckernel_names;
    }
    bool operator<(const CoreThreadInfo &rhs) const
    {
        return ckernel_names < rhs.ckernel_names;
    }
};
using CoreInfo = std::map<uint, CoreThreadInfo>;

struct BuildCoreImageResult
{
    std::map<uint, TriscImageBuildResult> trisc_images;
};

std::pair<TensixCoreImage, TriscImageAnalysis> load_and_analyse_trisc_image(
    const std::string &hex_file, bool allow_use_of_reserved_space, std::uint32_t trisc_size);
BuildCoreImageResult build_core_image(const std::string &testdir, const std::string &testname, const CoreInfo &core_info, const YAML::Node &ttx_manifest, std::vector<std::string> all_kernels, const std::string device);
void validate_core_build(xy_pair core, const BuildCoreImageResult &images);
void validate_trisc_image(const xy_pair core, const uint tid, const TriscImageBuildResult &image);
std::vector<std::string> get_all_kernels(std::string kernel_list_path);
std::string get_unused_kernels(std::vector<std::string> used_kernels, std::vector<std::string> all_kernels);

bool run_command(const std::string &cmd)
{
    status("running: `" + cmd + '`');
    int ret = system(cmd.c_str());
    return (ret == 0);
}

int main(int argc, char *argv[])
{
#if defined(BAZEL) || defined(TEST_BAZEL)
    status("This is CKTI, in Bazel mode.");
#else
    status("This is CKTI, in non-Bazel mode.");
#endif
    TCLAP::CmdLine cmd("C Kernel Test Image creator", '=');
    TCLAP::ValueArg<std::string>  test_name("", "test", "Test name", true, "", "string", cmd);
    TCLAP::ValueArg<std::string>  test_dir("", "dir", "Test output directory", true, "", "directory", cmd);
    TCLAP::ValueArg<std::string>  kernel_list_path("", "kernel_list_path", "Full list of kernel names", true, "", "string", cmd);
    TCLAP::ValueArg<std::string>  device_type("", "device", "Target device", true, "", "string", cmd);

    cmd.parse(argc, argv);
    
    std::string testname = test_name.getValue();
    std::string testdir = test_dir.getValue();
    std::string kernel_list_fullpath = kernel_list_path.getValue();
    std::string device = device_type.getValue();

    std::vector<std::string> all_kernels = get_all_kernels(kernel_list_fullpath);


    std::string filename = testdir + "/" + testname + ".ttx";

    CommandAssembler::ReadArchive arch(filename);
    CommandAssembler::SystemImage system_image = CommandAssembler::SystemImage::load(arch);

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

    std::vector<std::tuple<xy_pair, uint, CoreThreadInfo>> all_cores_and_threads_mutable;
    std::map<xy_pair, CoreInfo> cores_and_threads_mutable;
    std::vector<xy_pair> all_cores_mutable;
    for (const auto &core_and_info : ckernel_node["by_core"])
    {
        const auto core = core_and_info.first.as<xy_pair>();
        all_cores_mutable.push_back(core);
        cores_and_threads_mutable[core]; // ensure it exists
        for (const auto &thread_and_info : core_and_info.second["by_thread"])
        {
            const auto thread = thread_and_info.first.as<uint>();
            auto &curr = all_cores_and_threads_mutable.emplace_back(std::make_tuple(core, thread, CoreThreadInfo{}));
            cores_and_threads_mutable[core][thread]; // ensure it exists
            for (const auto &ckernel_name_and_info : thread_and_info.second["by_ckernel"])
            {
                std::get<2>(curr).ckernel_names.push_back(ckernel_name_and_info.first.as<std::string>());
                cores_and_threads_mutable[core][thread].ckernel_names.push_back(ckernel_name_and_info.first.as<std::string>());
            }
        }
    }
    const auto &all_cores_and_threads = all_cores_and_threads_mutable;
    const auto &cores_and_threads = cores_and_threads_mutable;

    const auto &all_cores = all_cores_mutable;

#if defined(BAZEL) || defined(TEST_BAZEL)
    //we copy here in order to create a distinct input
    const auto old_filename = filename;
    filename = filename.erase(filename.size()-4, filename.size()-1);
    filename += ".versim.ttx";
    std::string cp_command = "cp " + old_filename + " " + filename;
    if (!run_command(cp_command))
        error("Can't cp ttx into" + filename);
    if (!run_command("chmod u+w " + filename))
        error("Can't set u+w perms on ttx" + filename);
#endif

    CommandAssembler::WriteArchive wrarch(filename, true); // append

    const auto get_blank_image = [&](const xy_pair &core) -> const auto &
    {
        (void) core;
        static TensixCoreImage image;
        return image;
    };

    // Add ckernels to all cores
    std::for_each(all_cores.begin(), all_cores.end(), [&](xy_pair core) {
        if (not system_image.soc().GetNodeProperties(core).worker)
        {
            return; // effectively a 'continue'
        }

        if (cores_and_threads.find(core) == cores_and_threads.end())
        {
            error("No ckernel info in manifest for core " + format_node(core));
        }

        const auto &core_info = cores_and_threads.at(core);
        const std::string core_folder_name = format_node(core);
        status("Computing and adding ckernel-related items for core " + core_folder_name);

        //
        // merge and add ckernel images
        //

        const auto &images = build_core_image(testdir, testname, core_info, ttx_manifest, all_kernels, device);

        validate_core_build(core, images);

        TensixCoreImage merged_image;
        for (const auto &[tid, trisc_image] : images.trisc_images)
        {
            trisc_image.image.for_each_element([&](auto addr, auto val) { merged_image[addr] = val; });
        }

        std::stringstream binary_stream;
        CommandAssembler::binary_image_writer writer(binary_stream);
        merged_image.for_each_element(
            [&writer](CommandAssembler::memory::address_t address, CommandAssembler::memory::word_t value) { writer.add(address, value); });
        writer.end();

        const auto ckernel_bin_arch_path = core_folder_name + "/ckernels.bin";
        wrarch.write_archive_entry(ckernel_bin_arch_path, binary_stream.str());
        status("Added ckernel bin:" + ckernel_bin_arch_path);

        //
        // merge and add fwlogs
        //

        std::stringstream merged_ckernel_fwlog;
        for (const auto &[tid, trisc_image] : images.trisc_images)
        {
            std::ifstream fwlog_in(trisc_image.fwlog_file);
            if (not fwlog_in)
            {
                error("unable to read fwlog " + trisc_image.fwlog_file);
            }
            merged_ckernel_fwlog << fwlog_in.rdbuf();
        }

        const auto merged_fwlog_arch_path = core_folder_name + "/ckernel.fwlog";
        wrarch.write_archive_entry(merged_fwlog_arch_path, merged_ckernel_fwlog.str());
        status("Added ckernel fwlog:" + ckernel_bin_arch_path);
    });

    return 0;
}

BuildCoreImageResult build_core_image(const std::string &testdir, const std::string &testname, const CoreInfo &core_info, const YAML::Node &ttx_manifest, std::vector<std::string> all_kernels, const std::string device)
{
    // ***
    // Note: This function isn't really thread-safe, as there is nothing stopping it from building in the same directory as another thread.
    //       The 'cache' should be changed to use std::future
    // ***
    static std::mutex cache_mutex;
    static std::map<CoreInfo, BuildCoreImageResult> cache;

    // early-exit if it's already been computed
    { // scope for lock
        std::lock_guard cache_lock(cache_mutex);
        auto image_lookup = cache.find(core_info);
        if (image_lookup != cache.end())
        {
            return image_lookup->second;
        }
    }

    BuildCoreImageResult result;

    // build for each thread
    for (const auto &[thread_, thread_info] : core_info)
    {
        auto &thread = thread_;
        std::string kernel_list;
        std::vector<std::string> used_kernels_vector;
        for (const auto &kernel_name : thread_info.ckernel_names)
        {
            kernel_list += "c" + kernel_name + " ";
            used_kernels_vector.push_back(kernel_name);
        }
        std::string unused_kernels = get_unused_kernels(used_kernels_vector, all_kernels);

        statusf([&](auto &&s) { s << "Building a Trisc " << thread << " image with CKernels '" << kernel_list << '\''; });

        std::stringstream output_dir;
        output_dir << testdir << "/ckernels/tensix_thread" << thread << '-' << std::hex << std::hash<std::string>{}(kernel_list);

// life is all about compromising; bazel commands are root-relative, this is only set when building in bazel
#ifdef BAZEL
        //do we mean ckernels or test ckerenels?
        const auto make_args = "-f src/ckernels/Makefile.bzl";
#elif TEST_BAZEL
        //TODO: update this
        const auto make_args = "-f src/test_ckernels/Makefile.bzl";
#else 
        const auto make_args = "-C ..";
#endif

        std::string cli_output_dir = "\"" + output_dir.str() + "\"";
        boost::replace_all(cli_output_dir, ":", "\\:");

        std::stringstream make_cmd;
        make_cmd << " OUTPUT_DIR=" << cli_output_dir; // only want to set this for the first makefile
        // make_cmd << " OUTPUT_DIR=" << output_dir.str(); // only want to set this for the first makefile
        make_cmd << " make " << make_args;
        make_cmd << " TEST=" << testname;
        make_cmd << " FIRMWARE_NAME=tensix_thread" << thread;
        make_cmd << " KERNELS='" << kernel_list << '\'';
        make_cmd << " UNUSED_KERNELS='" << unused_kernels << '\'';
        make_cmd << " ARCH_NAME='" << device << '\'';

        const YAML::Node &genargs = ttx_manifest["genargs"];
        std::uint32_t trisc_thread_size = 16*1024;
        std::string trisc_map_key = "trisc" + std::to_string(thread) + "_size";

        if (genargs.IsDefined() || not genargs.IsNull())
        {
            for (const auto &genarg : genargs)
            {
                trisc_thread_size = genarg[trisc_map_key] ? genarg[trisc_map_key].as<uint32_t>() : trisc_thread_size;
                
                if (YAML::Node trisc0_size = genarg["trisc0_size"]) { make_cmd << " TRISC0_SIZE=" << trisc0_size; }
                else if (YAML::Node trisc1_size = genarg["trisc1_size"]) { make_cmd << " TRISC1_SIZE=" << trisc1_size; }
                else if (YAML::Node trisc2_size = genarg["trisc2_size"]) { make_cmd << " TRISC2_SIZE=" << trisc2_size; }
                else if (YAML::Node local_mem_en_node = genarg["local_memory_enable"]) { make_cmd << " TRISC_L0_EN=" << local_mem_en_node.as<int>(); }
            }
        }

        make_cmd << " LINKER_SCRIPT_NAME=trisc" << thread << ".ld";
        if (!run_command(make_cmd.str()))
        {
            errorf([&](auto &&s) { s << "Build failed for a thread " << thread << " with CKernels '" << kernel_list << '\''; });
        }

        const std::string hex_file = output_dir.str() + "/tensix_thread" + std::to_string(thread) + ".hex";

        // analyse resulting image
        const bool is_packer_thread = thread == 2;
        const bool allow_use_of_reserved_space = is_packer_thread;
        auto [image, analysis] = load_and_analyse_trisc_image(hex_file, allow_use_of_reserved_space, trisc_thread_size);
        result.trisc_images.emplace(thread,
            TriscImageBuildResult{
                .image = std::move(image),
                .hex_file = std::move(hex_file),
                .fwlog_file = output_dir.str() + "/ckernel.fwlog",
                .analysis = std::move(analysis),
            });
    }

    // check again, and insert if not present
    { // scope for lock
        std::lock_guard cache_lock(cache_mutex);
        auto image_lookup = cache.find(core_info);
        if (image_lookup != cache.end())
        {
            return image_lookup->second;
        }
        else
        {
            return cache.emplace(core_info, std::move(result)).first->second;
        }
    }
};

std::pair<TensixCoreImage, TriscImageAnalysis> load_and_analyse_trisc_image(
    const std::string &hex_file, bool allow_use_of_reserved_space, std::uint32_t trisc_size)
{
    const std::size_t max_size = trisc_size + (allow_use_of_reserved_space ? CommandAssembler::Kernels::RESERVED_SIZE : 0);

    TensixCoreImage image;
    image.add_hex_file(hex_file);

    return {
        std::move(image),
        {
            .max_allowed_size = max_size,
            .num_used_bytes = (image.max_address() - image.min_address()) * sizeof(TensixCoreImage::element_type),
        },
    };
}

void validate_core_build(const xy_pair core, const BuildCoreImageResult &images)
{
    const auto preamble = [&](auto &s) -> auto &
    {
        return s << "Core " << format_node(core) << ": ";
    };

    TensixCoreImage::address_type highest_limit_addresses = 0;
    for (const auto &[tid, trisc_image] : images.trisc_images)
    {
        validate_trisc_image(core, tid, trisc_image);
        highest_limit_addresses = std::max(highest_limit_addresses, trisc_image.image.max_address() * (int) sizeof(TensixCoreImage::element_type));
    }

    statusf([&](auto &&s) { preamble(s) << "first unused byte address after ckernel FW is 0x" << std::hex << highest_limit_addresses; });
}

void validate_trisc_image(const xy_pair core, const uint tid, const TriscImageBuildResult &image)
{
    const auto &ana = image.analysis;

    const auto preamble = [&](auto &s) -> auto &
    {
        return s << "Core " << format_node(core) << "'s Thread " << tid << ": ";
    };

    statusf([&](auto &&s) {
        preamble(s) << "is using 0x" << std::hex << ana.num_used_bytes << " of 0x" << std::hex << ana.max_allowed_size << " available bytes ("
                    << ana.num_used_bytes * 100.0 / ana.max_allowed_size << "%)";
    });

    if (ana.is_oversize())
    {
        errorf([&](auto &&s) {
            preamble(s) << "is using more than it's allocated space (using 0x" << std::hex << ana.num_used_bytes << " of 0x" << std::hex << ana.max_allowed_size
                        << " available bytes";
        });
    }
    else if (not ana.is_usable())
    {
        errorf([&](auto &&s) { preamble(s) << "is unusable, and there is no special message for this reason"; });
    }
}

std::vector<std::string> get_all_kernels(std::string kernel_list_path){
    YAML::Node kernel_list_yaml;
    std::vector<std::string> all_kernels;
    try
    {
        kernel_list_yaml = YAML::LoadFile(kernel_list_path);
    }
    catch (const std::exception &e)
    {
        errorf([&](auto &&s) { s << "Unable to load kernel list file: " << e.what(); });
    }
    for (const auto& kernel : kernel_list_yaml) {
        all_kernels.push_back(kernel.first.as<std::string>()); // prints Foo

        // const YAML::Node& value = kv.second;  // the value
    }

    return all_kernels;
}

std::string get_unused_kernels(std::vector<std::string> used_kernels, std::vector<std::string> all_kernels){

  std::unordered_set<std::string> used_kernel_set(used_kernels.begin(), used_kernels.end());
  std::unordered_set<std::string> unused_kernel_set(all_kernels.begin(), all_kernels.end());

  for (std::string kernel_name : used_kernel_set){
    unused_kernel_set.erase(kernel_name);
  }

  std::string unused_kernel_string;
  for(std::string kernel : unused_kernel_set){
    unused_kernel_string += "c" + kernel + " ";
  }
  return unused_kernel_string;
}
