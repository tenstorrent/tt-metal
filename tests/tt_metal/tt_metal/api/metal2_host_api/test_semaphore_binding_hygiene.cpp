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
// undercount and AUTO may choose the wrong mechanism for a semaphore that actually has
// concurrent writers.
//
// The stakes rose once AUTO gained the ability to auto-select DM_LOCAL_CACHED: a cached
// semaphore is RELOCATED into the cached-only pool (noc_semaphore.h sem_l1_offset()), so a
// declared binder and an undeclared toucher address two DIFFERENT words -- the semaphore
// SPLITS and a wait() never completes, rather than merely losing an update as it would
// under LOCAL_NONATOMIC/EXTERNAL (both of which stay in the kernel_config ring). Keeping
// this lint green is therefore load-bearing for the cached pick.
//
// An audit found ZERO production Metal 2.0 kernels using the raw path. This test keeps
// it that way: it fails if a Metal 2.0 kernel source starts using raw semaphore access.
//
// SCOPE / LIMITS (deliberate, be honest about them):
//  - LEGACY kernels are not checked and do not matter: they never reach the AUTO path
//    (it lives in MakeProgramFromSpec), and they manage their own CreateSemaphore words.
//  - This is a source lint, not a proof. It can be defeated by macros, helper functions,
//    or an address handed in as a plain runtime arg, so it is a regression guard rather
//    than a soundness proof. The cached pick does not rest on it alone: the classifier
//    independently requires Gen2, a single-cell semaphore, every binder on that node, and
//    exactly ONE binder kernel.
//  - The detector itself is covered by DetectorFlagsKnownViolations below; without that
//    positive control this sweep could go silently vacuous.
// ============================================================================

// A Metal 2.0 kernel is one that uses the generated accessors/args.
bool looks_like_metal2_kernel(const std::string& text) {
    return text.find("get_arg(args::") != std::string::npos || text.find("sem::") != std::string::npos ||
           text.find("dfb::") != std::string::npos;
}

// Strip comments so prose mentioning a pattern (e.g. the explanatory comments in the keystone
// kernels) is not flagged as code. Tracks /* */ state properly: an earlier version skipped any line
// whose first non-space character was '*', which also silently dropped real statements such as
//     *(volatile uint32_t*)get_semaphore(3) += 1;
// i.e. it hid exactly the raw accesses this lint exists to find.
std::string strip_comments(const std::string& text) {
    std::string out;
    out.reserve(text.size());
    bool in_block = false;
    bool in_line_comment = false;
    for (size_t i = 0; i < text.size(); i++) {
        if (in_line_comment) {
            if (text[i] == '\n') {
                in_line_comment = false;
                out += '\n';
            }
            continue;
        }
        if (in_block) {
            if (text[i] == '*' && i + 1 < text.size() && text[i + 1] == '/') {
                in_block = false;
                i++;
            } else if (text[i] == '\n') {
                out += '\n';
            }
            continue;
        }
        if (text[i] == '/' && i + 1 < text.size() && text[i + 1] == '/') {
            in_line_comment = true;
            continue;
        }
        if (text[i] == '/' && i + 1 < text.size() && text[i + 1] == '*') {
            in_block = true;
            i++;
            continue;
        }
        out += text[i];
    }
    return out;
}

// Raw (undeclared) semaphore access patterns. Each is matched on its OWN, not as a literal pair:
// an earlier version only flagged "noc_semaphore_inc(get_noc_addr(" and so missed the (very common)
// two-statement form where the address is computed on a previous line -- which let a real in-tree
// raw writer through. A declared binding never needs any of these: it uses Semaphore(sem::<name>),
// whose up()/down() call into these primitives from inside the framework header, not kernel source.
// A baked sem:: token must be constructed WITHOUT an explicit template argument list, so CTAD can
// adopt the host-chosen scope. `Semaphore<> s(sem::x)` (or any explicit <...>) pins the class default
// LOCAL_NONATOMIC instead, contradicting the baked scope and firing the token ctor's static_assert --
// a hard JIT compile failure. This is not a "raw access", so the pattern list below cannot see it;
// six live sites reached the tree that way, four of them for weeks. Detected separately here.
std::vector<std::string> find_pinned_scope_constructions(const std::string& code) {
    std::vector<std::string> found;
    size_t pos = 0;
    while ((pos = code.find("Semaphore<", pos)) != std::string::npos) {
        const size_t close = code.find('>', pos);
        if (close == std::string::npos) {
            break;
        }
        // Only a construction over a sem:: token matters; `Semaphore<> s(raw_id)` uses the unaffected
        // uint32_t ctor, and a `Semaphore<...>&` parameter is fine as long as it is scope-generic.
        const size_t stmt_end = code.find(';', close);
        if (stmt_end != std::string::npos && code.compare(close, stmt_end - close, code.substr(close, stmt_end - close)) == 0) {
            const std::string stmt = code.substr(close, stmt_end - close);
            if (stmt.find("sem::") != std::string::npos) {
                found.emplace_back("Semaphore<...> constructed over a sem:: token (drop <> and let CTAD "
                                   "adopt the baked scope)");
            }
        }
        pos = close + 1;
    }
    return found;
}

std::vector<std::string> find_raw_semaphore_uses(const std::string& code) {
    static const char* kPatterns[] = {
        "get_semaphore(",                // turning a semaphore id into a raw address
        "noc_semaphore_inc(",            // atomic increment at a raw address (local or remote)
        "noc_semaphore_set(",            // raw set
        "noc_semaphore_set_remote(",     // remote set
        "noc_semaphore_set_multicast(",  // multicast set
        "noc_semaphore_inc_multicast(",  // multicast increment
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
    // Hardware keystone probes: these deliberately issue raw NoC atomics at SCRATCH L1 addresses
    // (never at a declared semaphore) in order to MEASURE hardware behaviour -- the
    // self-targeted-atomic keystone, the DM cache-line-width probe, and the NoC atomic-opcode probe.
    // Raw access is the thing under test, so they must stay raw.
    if (path.find("dm_cacheline_probe.cpp") != std::string::npos ||
        path.find("noc_self_atomic.cpp") != std::string::npos ||
        path.find("noc_atomic_ops_probe.cpp") != std::string::npos) {
        return true;
    }
    // Residency instrumentation: sem_census_probe deliberately reads its OWN declared semaphore's
    // kernel_config RING slot (read-only, via the uncached alias) so a test can prove that a cached
    // semaphore's count really lives in the pool. It adds no undeclared writer.
    if (path.find("sem_census_probe.cpp") != std::string::npos) {
        return true;
    }
    // Watcher fault injection: this kernel issues a raw multicast atomic at an INVALID range,
    // targeting a data-buffer address, specifically to trip the watcher's sanitizer. Its host test
    // (debug_tools/watcher/test_sanitize.cpp) declares no SemaphoreSpec at all, so no semaphore --
    // declared or otherwise -- is involved; the malformed op is the thing under test.
    if (path.find("dram_copy_to_noc_coord_2_0.cpp") != std::string::npos) {
        return true;
    }
    return false;
}

// Kernel sources worth scanning: the metal test kernels and the TT-NN op kernels.
bool is_kernel_source(const std::filesystem::path& p) {
    if (p.extension() != ".cpp") {
        return false;
    }
    const std::string s = p.string();
    return s.find("test_kernels") != std::string::npos || s.find("/kernels/") != std::string::npos;
}

// POSITIVE CONTROL for the detector itself. Without this, the scan test above is
// "always green by construction": if strip_comments or the pattern list silently stopped matching,
// the sweep would still report zero violations and pass. (That is not hypothetical -- an earlier
// version dropped '*'-leading statement lines and only matched noc_semaphore_inc when it was
// literally followed by get_noc_addr, which let a real in-tree raw writer through.)
TEST(Metal2SemaphoreHygiene, DetectorFlagsKnownViolations) {
    struct Case {
        const char* name;
        std::string body;
        bool should_flag;
    };
    const std::vector<Case> cases = {
        {"id_to_address", "void kernel_main() { uint32_t a = get_semaphore(get_arg_val<uint32_t>(0)); }", true},
        // Must survive comment-stripping: the statement starts with '*'.
        {"deref_leading_star", "void kernel_main() {\n    *(volatile uint32_t*)get_semaphore(3) += 1;\n}", true},
        // Address computed on a PREVIOUS line -- the two-statement form that used to escape.
        {"split_noc_poke",
         "void kernel_main() {\n    uint64_t a = get_noc_addr(1, 1, 64);\n    noc_semaphore_inc(a, 1);\n}",
         true},
        {"raw_local_set", "void kernel_main() { noc_semaphore_set(ptr, 5); }", true},
        // The shape that slipped past this lint six times: explicit <> pins the class-default scope.
        {"pinned_scope_ctor", "void kernel_main() { Semaphore<> s(sem::counter); }", true},
        // ...but the same spelling over a RAW id uses the unaffected uint32_t ctor and is legal.
        {"pinned_scope_raw_id_ok", "void kernel_main() { Semaphore<> s(sem_id); }", false},
        {"multicast_set", "void kernel_main() { noc_semaphore_set_multicast(a, b, 1); }", true},
        // A declared binding: the managed accessor only, no raw primitive in kernel source.
        {"clean_declared", "void kernel_main() {\n    Semaphore s(sem::counter);\n    s.up(1);\n}", false},
        // Prose mentioning the patterns must NOT be flagged (line and block comments).
        {"comment_only_line", "// calls get_semaphore(id) and noc_semaphore_inc(addr, 1)\nvoid kernel_main() {}", false},
        {"comment_only_block",
         "/*\n * uses get_semaphore(x)\n * and noc_semaphore_inc(y, 1)\n */\nvoid kernel_main() {}",
         false},
    };

    for (const auto& c : cases) {
        auto found = find_raw_semaphore_uses(strip_comments(c.body));
        for (auto& p : find_pinned_scope_constructions(strip_comments(c.body))) {
            found.push_back(p);
        }
        if (c.should_flag) {
            EXPECT_FALSE(found.empty()) << "detector MISSED a raw semaphore access in case '" << c.name
                                        << "' -- the scan test would silently pass on this code";
        } else {
            EXPECT_TRUE(found.empty()) << "detector FALSE-POSITIVED on case '" << c.name << "' (first: "
                                       << (found.empty() ? std::string{} : found.front()) << ")";
        }
    }
}

TEST(Metal2SemaphoreHygiene, NoRawSemaphoreAccessInMetal2Kernels) {
    // Locate the repo WITHOUT depending on the environment. This test used to GTEST_SKIP when
    // TT_METAL_HOME was unset, which is the worst failure mode for a lint: it protects nothing while
    // still reporting success. __FILE__ is this source's own path, so the root is derivable from it;
    // TT_METAL_HOME is only a fallback for an out-of-tree layout.
    std::filesystem::path root;
    {
        const std::string self{__FILE__};
        const std::string marker{"/tests/tt_metal/"};
        const size_t at = self.find(marker);
        if (at != std::string::npos) {
            root = std::filesystem::path{self.substr(0, at)};
        }
    }
    if (root.empty() || !std::filesystem::exists(root / "tests")) {
        const char* home = std::getenv("TT_METAL_HOME");
        ASSERT_NE(home, nullptr) << "cannot locate the repo from __FILE__ (" << __FILE__
                                 << ") and TT_METAL_HOME is unset -- this lint would otherwise pass "
                                    "vacuously without scanning anything";
        root = std::filesystem::path{home};
    }
    ASSERT_TRUE(std::filesystem::exists(root / "tests")) << "derived repo root has no tests/ dir: " << root;
    // Broad roots: "tests" (not just tests/tt_metal -- the TT-NN test tree also holds Metal 2.0
    // kernels, e.g. under unit_tests/gtests/accessor/kernels/) and the whole op tree. The per-file
    // predicate below narrows to kernel sources.
    const std::vector<std::filesystem::path> scan_roots = {
        root / "tests",
        root / "ttnn",
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
            const std::string code = strip_comments(text);
            for (const auto& pat : find_raw_semaphore_uses(code)) {
                violations.push_back(path_str + "  uses raw: " + pat);
            }
            for (const auto& pat : find_pinned_scope_constructions(code)) {
                violations.push_back(path_str + "  pins scope: " + pat);
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
