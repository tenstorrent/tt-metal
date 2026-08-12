// SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

// A plain-C window onto tt-metal's host-side packed kernel image, so metal.py can
// inject detours with ctypes and no tt-metal source change. The image is the one
// slow dispatch re-writes into L1 on every launch:
//   LaunchProgram -> ConfigureDeviceWithProgram -> ConfigureKernelGroup
//                 -> ComputeKernel::configure -> llrt::write_binary_to_address
// which is why the perturbation goes here and not into L1. See metal.py.
//
// WHY A const_cast IS SAFE HERE
// -----------------------------
// llrt::get_risc_binary caches exactly one image per ELF path, process-wide and
// permanently (the function-local `cache` in llrt/llrt.cpp), and hands it back
// const. For normal callers that immutability is the contract; for this tool it is
// the mechanism -- one object per kernel means mutating it in place perturbs every
// subsequent launch, and there is no second copy to keep in sync. The cast is
// confined to this file, and this file is only ever loaded by the perturbation tool.
//
// Going through the public data() accessor rather than poking the object's bytes
// from ctypes keeps us off std::vector's internal layout: the compiler builds the
// std::string and std::function arguments, so there is no ABI guesswork on the
// Python side and the seam is a plain pointer.

#include <cstdint>
#include <functional>
#include <string>
#include <vector>

// Only tt_memory.h -- not llrt.hpp. The latter pulls core_coord / fmt / umd and
// forces a metal-sized include tree onto this tiny shim. The linker resolves
// get_risc_binary from libtt_metal.so; the declaration below must match
// llrt/llrt.hpp exactly.
#include "llrt/tt_memory.h"

namespace tt::llrt
{
const ll_api::memory& get_risc_binary(
    const std::string& path,
    ll_api::memory::Loading loading                             = ll_api::memory::Loading::DISCRETE,
    const std::function<void(ll_api::memory&)>& update_callback = nullptr);
}

extern "C"
{
    // Hand back the words of the packed image that metal writes to L1 for `elf_path`.
    //
    // `loading` must match the HAL's memory_load for that processor or the cached image
    // will not be the one metal uses -- TRISC compute is CONTIGUOUS_XIP (2) on both
    // Wormhole and Blackhole; Wormhole NCRISC is plain CONTIGUOUS (1). It is passed in
    // rather than assumed so a future reader/writer actor cannot silently create a
    // second, wrongly-loaded cache entry. `*out_loading` reports what the cache actually
    // holds so the caller can assert on a hit.
    //
    // The text segment is packed first, so word 0 of the returned buffer is the first
    // word of .text and `*out_text_words` bounds that region. `*out_text_addr` is the
    // image's own idea of where text lives: XIPify rewrites it to 0, so a non-zero value
    // means this image was never XIPified and the caller's offsets would be wrong.
    //
    // The returned pointer is valid for the life of the process (the cache never evicts),
    // so the caller may hold it for a whole sweep. Returns nullptr on failure.
    std::uint32_t* ttnop_image_words(
        const char* elf_path,
        std::uint32_t loading,
        std::uint32_t* out_total_words,
        std::uint32_t* out_text_words,
        std::uint32_t* out_loading,
        std::uint32_t* out_text_addr)
    {
        try
        {
            const std::string path(elf_path);
            // Named empty callback: a defaulted temporary here trips gcc
            // -Wdangling-reference on the returned cache reference (false positive --
            // the cache owns the image for the process lifetime).
            static const std::function<void(ll_api::memory&)> no_update;
            const ll_api::memory& image = tt::llrt::get_risc_binary(path, static_cast<ll_api::memory::Loading>(loading), no_update);

            // The one const_cast: see the file header. data() is the public accessor, so
            // this is a constness cast only, not a reinterpretation of the object.
            std::vector<std::uint32_t>& words = const_cast<std::vector<std::uint32_t>&>(image.data());

            *out_total_words = static_cast<std::uint32_t>(words.size());
            *out_text_words  = image.get_text_size() / sizeof(std::uint32_t);
            *out_loading     = static_cast<std::uint32_t>(image.get_loading());
            *out_text_addr   = image.get_text_addr();
            return words.data();
        }
        catch (...)
        {
            // Bad path, or called before the device is open so MetalContext is not up yet.
            // metal.py turns nullptr into a loud setup error, never a silent clean sweep.
            return nullptr;
        }
    }

} // extern "C"
