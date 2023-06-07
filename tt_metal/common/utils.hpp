#pragma once

#include <string>
#include "assert.hpp"

using std::string;
using std::cout;

#include <filesystem>
namespace fs = std::filesystem;

namespace tt
{
namespace utils
{
    bool run_command(const string &cmd, const string &log_file, const bool verbose);
    void create_file(string file_path_str);
    std::string get_root_dir();
    const std::string &get_reports_dir();

    // Ripped out of boost for std::size_t so as to not pull in bulky boost dependencies
    template <typename SizeT>
    inline void hash_combine(SizeT& seed, const SizeT value) {
        seed ^= value + 0x9e3779b9 + (seed << 6) + (seed >> 2);
    }

    template <class T>
    struct vector_hash {
    inline std::size_t operator()(const std::vector<T> &vec) const
    {
        size_t seed = vec.size();
        for(auto& i : vec) {
            hash_combine(seed, static_cast<std::size_t>(i));
        }
        return seed;
    }
};
}
}
