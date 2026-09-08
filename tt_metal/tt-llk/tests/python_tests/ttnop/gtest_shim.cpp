// SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
// SPDX-License-Identifier: Apache-2.0

// Patch a native gtest's process-local kernel image as it is loaded. Python
// still owns scanning and cave layout, so this shim only copies prepared words.

#include <dlfcn.h>

#include <cstdint>
#include <cstdlib>
#include <fstream>
#include <functional>
#include <mutex>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include "llrt/tt_memory.h"

namespace
{

constexpr const char* SYMBOL =
    "_ZN2tt4llrt15get_risc_binaryERKNSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEEEN6ll_api6memory7LoadingERKSt8functionIFvRSA_EE";
std::mutex mutex;

struct Patch
{
    std::string elf;
    std::uint32_t text_start = 0;
    std::vector<std::pair<std::uint32_t, std::uint32_t>> words;
};

Patch read_patch()
{
    const char* path = std::getenv("TTNOP_GTEST_PATCH");
    if (!path)
    {
        return {};
    }
    std::ifstream input(path);
    Patch patch;
    std::string address;
    std::string word;
    std::string text_start;
    if (!input || !std::getline(input, patch.elf) || !std::getline(input, text_start))
    {
        throw std::runtime_error("ttnop: invalid gtest patch file");
    }
    patch.text_start = std::stoul(text_start, nullptr, 0);
    while (input >> address >> word)
    {
        patch.words.emplace_back(std::stoul(address, nullptr, 0), std::stoul(word, nullptr, 0));
    }
    if (!input.eof() || patch.words.empty())
    {
        throw std::runtime_error("ttnop: empty or malformed gtest patch");
    }
    return patch;
}

void apply_patch(const std::string& path, ll_api::memory::Loading loading, const ll_api::memory& image)
{
    static const Patch patch = read_patch();
    if (patch.elf.empty() || patch.elf != path)
    {
        return;
    }
    if (loading != ll_api::memory::Loading::CONTIGUOUS_XIP || image.get_text_addr() != 0)
    {
        throw std::runtime_error("ttnop: selected kernel is not a zero-based XIP image");
    }

    // The dump keeps the ELF VMA, but the packed XIP image begins at zero.
    std::lock_guard lock(mutex);
    auto& data                     = const_cast<std::vector<std::uint32_t>&>(image.data());
    const std::uint32_t text_words = image.get_text_size() / 4;
    for (const auto& [address, word] : patch.words)
    {
        if (address < patch.text_start || (address - patch.text_start) % 4)
        {
            throw std::runtime_error("ttnop: invalid patch address");
        }
        const std::uint32_t index = (address - patch.text_start) / 4;
        if (index >= text_words || index >= data.size())
        {
            throw std::runtime_error("ttnop: patch falls outside kernel text");
        }
        data[index] = word;
    }

    const char* ack = std::getenv("TTNOP_GTEST_ACK");
    std::ofstream output(ack ? ack : "", std::ios::app);
    if (!output)
    {
        throw std::runtime_error("ttnop: cannot acknowledge gtest patch");
    }
    output << path << '\n';
}

} // namespace

namespace tt::llrt
{

const ll_api::memory& get_risc_binary(const std::string& path, ll_api::memory::Loading loading, const std::function<void(ll_api::memory&)>& callback)
{
    using Function             = decltype(&get_risc_binary);
    static const Function real = []
    {
        dlerror();
        auto* symbol = dlsym(RTLD_NEXT, SYMBOL);
        if (const char* error = dlerror())
        {
            throw std::runtime_error(std::string("ttnop: cannot resolve get_risc_binary: ") + error);
        }
        return reinterpret_cast<Function>(symbol);
    }();

    const ll_api::memory& image = real(path, loading, callback);
    if (const char* trace = std::getenv("TTNOP_GTEST_TRACE"))
    {
        std::lock_guard lock(mutex);
        std::ofstream(trace, std::ios::app) << path << '\t' << static_cast<unsigned>(loading) << '\n';
    }
    apply_patch(path, loading, image);
    return image;
}

} // namespace tt::llrt
