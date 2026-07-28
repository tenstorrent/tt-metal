// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <filesystem>
#include <fstream>
#include <sstream>
#include <string>
#include <string_view>
#include <vector>
#include <gtest/gtest.h>

namespace tt::tt_metal {
namespace {

// ============================================================================
// Metal 2.0 semaphore-binding hygiene (source lint; no device needed).
// ============================================================================
//
// WHY THIS EXISTS
// ---------------
// The AUTO semaphore-scope classifier (ResolveSemaphoreScope) picks each semaphore's
// physical mechanism -- cheap non-atomic LOCAL_NONATOMIC vs atomic EXTERNAL -- from a
// census of the kernels that DECLARE a binding to it (KernelSpec::semaphore_bindings,
// with a per-binding AccessType). That census is only complete if every Metal 2.0
// kernel reaches its semaphores through the managed accessor:
//
//     Semaphore s(sem::<name>);      // declared -> the host sees this writer
//
// A kernel can also reach a semaphore RAW, bypassing the declaration:
//
//     get_semaphore(get_arg_val<uint32_t>(i))              // id -> address, undeclared
//     noc_semaphore_inc(get_noc_addr(x, y, addr), 1)       // poke a peer's word
//     noc_semaphore_set_remote(...)
//
// A raw access is an UNDECLARED WRITER: the host cannot see it, so the census can
// undercount and AUTO may choose the cheap non-atomic mechanism for a semaphore that
// actually has concurrent writers. (That is not a regression -- before AUTO, every
// semaphore was non-atomic anyway -- but it caps what AUTO can guarantee, and it is the
// reason AUTO must never auto-select the DM_LOCAL_CACHED cached fast path.)
//
// An audit found ZERO production Metal 2.0 kernels using the raw path. This test keeps
// it that way: it fails if a Metal 2.0 kernel source starts using raw semaphore access.
//
// SCOPE / LIMITS (deliberate, be honest about them):
//  - LEGACY kernels are not checked and do not matter: they never reach the AUTO path
//    (it lives in MakeProgramFromSpec), and they manage their own CreateSemaphore words.
//  - This is a source lint, not a proof. It can be defeated by macros, helper functions,
//    or an address handed in as a plain runtime arg. It catches accidental regressions;
//    it is NOT sufficient on its own to justify auto-selecting DM_LOCAL_CACHED.
// ============================================================================

// A Metal 2.0 kernel is one that uses the generated accessors/args.
bool looks_like_metal2_kernel(const std::string& text) {
    return text.find("get_arg(args::") != std::string::npos || text.find("sem::") != std::string::npos ||
           text.find("dfb::") != std::string::npos;
}

// Strip // comments and obvious block-comment continuation lines, so prose mentioning a
// pattern (e.g. explanatory comments in the keystone kernels) is not flagged as code.
std::string strip_comments(const std::string& text) {
    std::istringstream in(text);
    std::ostringstream out;
    std::string line;
    while (std::getline(in, line)) {
        const size_t first = line.find_first_not_of(" \t");
        if (first != std::string::npos && (line.compare(first, 2, "//") == 0 || line[first] == '*')) {
            continue;  // whole-line comment, or a line inside a block comment
        }
        const size_t slashes = line.find("//");
        out << (slashes == std::string::npos ? line : line.substr(0, slashes)) << '\n';
    }
    return out.str();
}

// Raw (undeclared) semaphore access patterns.
std::vector<std::string> find_raw_semaphore_uses(const std::string& code) {
    static const char* kPatterns[] = {
        "get_semaphore(",             // turning an id into an address directly
        "noc_semaphore_inc(get_noc_addr(",  // atomic poke at a computed address
        "noc_semaphore_set_remote(",  // remote set
    };
    std::vector<std::string> found;
    for (const char* pat : kPatterns) {
        if (code.find(pat) != std::string::npos) {
            found.emplace_back(pat);
        }
    }
    return found;
}

// Files intentionally exempt, with the reason. Keep this list MINIMAL: an entry here can
// mask a real regression in that file.
bool is_allowlisted(const std::string& path) {
    // Hardware keystone probes: these deliberately issue raw NoC atomics at SCRATCH L1
    // addresses (not at any declared semaphore) to measure hardware behaviour -- the
    // self-targeted-atomic keystone and the DM cache-line-width probe. They must stay raw;
    // that is the thing under test.
    return path.find("dm_cacheline_probe.cpp") != std::string::npos ||
           path.find("noc_self_atomic.cpp") != std::string::npos;
}

// Kernel sources worth scanning: the metal test kernels and the ttnn op kernels.
bool is_kernel_source(const std::filesystem::path& p) {
    if (p.extension() != ".cpp") {
        return false;
    }
    const std::string s = p.string();
    return s.find("test_kernels") != std::string::npos || s.find("/kernels/") != std::string::npos;
}

TEST(Metal2SemaphoreHygiene, NoRawSemaphoreAccessInMetal2Kernels) {
    const char* home = std::getenv("TT_METAL_HOME");
    if (home == nullptr) {
        GTEST_SKIP() << "TT_METAL_HOME not set; cannot locate kernel sources to lint";
    }
    const std::filesystem::path root{home};
    const std::vector<std::filesystem::path> scan_roots = {
        root / "tests" / "tt_metal",
        root / "ttnn" / "cpp",
    };

    std::vector<std::string> violations;
    uint32_t metal2_scanned = 0;
    uint32_t files_scanned = 0;

    for (const auto& scan_root : scan_roots) {
        if (!std::filesystem::exists(scan_root)) {
            continue;
        }
        for (const auto& entry : std::filesystem::recursive_directory_iterator(scan_root)) {
            if (!entry.is_regular_file() || !is_kernel_source(entry.path())) {
                continue;
            }
            files_scanned++;
            std::ifstream f(entry.path());
            if (!f) {
                continue;
            }
            std::stringstream buf;
            buf << f.rdbuf();
            const std::string text = buf.str();
            if (!looks_like_metal2_kernel(text)) {
                continue;  // legacy kernel: never reaches the AUTO path
            }
            metal2_scanned++;
            const std::string path_str = entry.path().string();
            if (is_allowlisted(path_str)) {
                continue;
            }
            for (const auto& pat : find_raw_semaphore_uses(strip_comments(text))) {
                violations.push_back(path_str + "  uses raw: " + pat);
            }
        }
    }

    EXPECT_GT(files_scanned, 0u) << "lint scanned no kernel sources -- check the scan roots";
    EXPECT_GT(metal2_scanned, 0u) << "lint found no Metal 2.0 kernels -- the detector may be stale";

    std::string report;
    for (const auto& v : violations) {
        report += "\n  " + v;
    }
    EXPECT_TRUE(violations.empty())
        << "A Metal 2.0 kernel reaches a semaphore RAW, bypassing its declared binding. The host's "
           "who-touches census (which drives the AUTO scope classifier) cannot see such a writer, so AUTO "
           "may pick the cheap non-atomic mechanism for a semaphore that actually has concurrent writers."
           "\n\nFix: declare a KernelSpec::SemaphoreBinding and use the managed accessor instead:"
           "\n    Semaphore s(sem::<accessor_name>);   // scope is host-baked, picked automatically"
           "\nIf the raw access is genuinely required (e.g. a hardware probe on scratch L1, not on a "
           "declared semaphore), add the file to is_allowlisted() with a reason."
           "\n\nOffending files:"
        << report;
}

}  // namespace
}  // namespace tt::tt_metal
