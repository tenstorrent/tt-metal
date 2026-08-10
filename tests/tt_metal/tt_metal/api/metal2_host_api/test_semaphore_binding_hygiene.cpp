// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <algorithm>
#include <cctype>
#include <cstdint>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <map>
#include <memory>
#include <sstream>
#include <string>
#include <vector>
#include <gtest/gtest.h>

namespace tt::tt_metal {
namespace {

// ============================================================================
// Metal 2.0 semaphore-binding hygiene (source lint; no device needed).
// ============================================================================
//
// The AUTO scope classifier (ResolveSemaphoreScope) picks each semaphore's physical mechanism
// from a census of the kernels that DECLARE a binding to it (KernelSpec::semaphore_bindings).
// A raw access -- get_semaphore(id), noc_semaphore_inc(addr, 1), ... -- is an UNDECLARED
// writer the host cannot see: AUTO may then pick a non-atomic mechanism for a semaphore
// that actually has concurrent writers. Worse under DM_LOCAL_CACHED: a cached semaphore is
// RELOCATED into the cached-only pool (noc_semaphore.h sem_l1_offset()), so a declared binder
// and an undeclared toucher address two DIFFERENT words -- the semaphore splits and a wait()
// never completes. Keeping this lint green is load-bearing for the cached pick.
//
// LIMITS (deliberate):
//  - LEGACY kernels are not checked: they never reach the AUTO path and manage their own
//    CreateSemaphore words.
//  - This is a regression guard, not a soundness proof: macros, helper functions, or an
//    address handed in as a runtime arg defeat it. The cached pick does not rest on it alone.
//  - DetectorFlagsKnownViolations below is the positive control; without it this sweep
//    could go silently vacuous.
// ============================================================================

// A Metal 2.0 kernel is one that uses the generated accessors/args. All five generated
// binding namespaces count (genfiles.cpp emits args/dfb/sem/tensor/scratch): a kernel that
// binds only tensors or scratchpads is still Metal 2.0 and must not touch semaphores raw.
bool looks_like_metal2_kernel(const std::string& text) {
    return text.find("get_arg(args::") != std::string::npos || text.find("sem::") != std::string::npos ||
           text.find("dfb::") != std::string::npos || text.find("tensor::") != std::string::npos ||
           text.find("scratch::") != std::string::npos;
}

// Strip comments and string/char literal contents so a banned pattern in prose or a string is
// not flagged as code, and a "//" or "/*" inside a string does not start a comment.
// LIMIT: raw string literals (R"(...)" ) are not modeled; kernels use none today.
std::string strip_comments(const std::string& text) {
    std::string out;
    out.reserve(text.size());
    bool in_block = false;
    bool in_line_comment = false;
    bool in_string = false;
    bool in_char = false;
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
        if (in_string || in_char) {
            if (text[i] == '\\') {
                i++;  // skip the escaped character
            } else if (text[i] == (in_string ? '"' : '\'')) {
                in_string = in_char = false;
                out += text[i];
            } else if (text[i] == '\n') {
                out += '\n';  // unterminated literal: fail safe at end of line
                in_string = in_char = false;
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
        if (text[i] == '"') {
            in_string = true;
            out += text[i];
            continue;
        }
        // Not after an alnum: 1'000 is a digit separator, not a char literal.
        if (text[i] == '\'' && (out.empty() || !std::isalnum(static_cast<unsigned char>(out.back())))) {
            in_char = true;
            out += text[i];
            continue;
        }
        out += text[i];
    }
    return out;
}

// A sem:: token must be constructed with NO template argument list, so CTAD adopts the baked
// scope and access rights. An explicit <...> pins the class defaults instead: it compiles until
// the host bakes something else, then fails at JIT (token-ctor static_assert). Not a raw access,
// so the pattern list below cannot see it; detected separately here.
std::vector<std::string> find_pinned_scope_constructions(const std::string& code) {
    std::vector<std::string> found;
    size_t pos = 0;
    while ((pos = code.find("Semaphore<", pos)) != std::string::npos) {
        // First '>' is a safe bound: no nested Semaphore<...<...>> spelling exists, and erring
        // short only over-flags.
        const size_t close = code.find('>', pos);
        if (close == std::string::npos) {
            break;
        }
        // Only a construction over a sem:: token matters: raw-id ctors are covered by
        // find_unbound_semaphore_constructions, and a scope-generic `Semaphore<...>&` parameter
        // is fine. Discriminate by the first non-whitespace character after '>': a reference/
        // pointer/list-position marker means a PARAMETER (skip); anything else is a potential
        // construction, scanned to the ';' so brace-init and line-wrapped spellings stay covered.
        size_t after = close + 1;
        while (after < code.size() && std::isspace(static_cast<unsigned char>(code[after]))) {
            after++;
        }
        if (after < code.size() && (code[after] == '&' || code[after] == '*' || code[after] == ',' ||
                                    code[after] == ')' || code[after] == ':')) {
            pos = close + 1;
            continue;
        }
        const size_t stmt_end = code.find(';', close);
        if (stmt_end != std::string::npos) {
            const std::string stmt = code.substr(close, stmt_end - close);
            if (stmt.find("sem::") != std::string::npos) {
                found.emplace_back(
                    "Semaphore<...> constructed over a sem:: token (drop <> and let CTAD "
                    "adopt the baked scope and access rights)");
            }
        }
        pos = close + 1;
    }
    return found;
}

// Raw-id Semaphore construction: `Semaphore s(x)` / `Semaphore<...> s{x}` where the constructor
// argument is not a sem:: token. Still an UNDECLARED census participant: the host cannot see it,
// so AUTO can pick a non-atomic mechanism out from under it. LIMITS: unnamed forms
// (`auto s = Semaphore(x)`, `new Semaphore(x)`, temporaries) are not modeled -- no kernel
// source uses them today.
std::vector<std::string> find_unbound_semaphore_constructions(const std::string& code) {
    std::vector<std::string> found;
    size_t pos = 0;
    while ((pos = code.find("Semaphore", pos)) != std::string::npos) {
        const size_t start = pos;
        pos += 9;  // strlen("Semaphore")
        // Word boundaries: SemaphoreBindingToken, MySemaphore, Semaphores... are not this class.
        if (start > 0) {
            const char before = code[start - 1];
            if (std::isalnum(static_cast<unsigned char>(before)) || before == '_') {
                continue;
            }
        }
        size_t p = pos;
        if (p < code.size() && (std::isalnum(static_cast<unsigned char>(code[p])) || code[p] == '_')) {
            continue;
        }
        // Optional template argument list; first '>' is a safe bound (see the pinned detector).
        if (p < code.size() && code[p] == '<') {
            const size_t close = code.find('>', p);
            if (close == std::string::npos) {
                break;
            }
            p = close + 1;
        }
        // Construction shape: exactly one identifier (the variable name), then '(' or '{'.
        // Anything else -- a reference/pointer parameter, a bare type mention -- is not one.
        while (p < code.size() && std::isspace(static_cast<unsigned char>(code[p]))) {
            p++;
        }
        const size_t id_begin = p;
        while (p < code.size() && (std::isalnum(static_cast<unsigned char>(code[p])) || code[p] == '_')) {
            p++;
        }
        if (p == id_begin) {
            continue;
        }
        while (p < code.size() && std::isspace(static_cast<unsigned char>(code[p]))) {
            p++;
        }
        if (p >= code.size() || (code[p] != '(' && code[p] != '{')) {
            continue;
        }
        const size_t stmt_end = code.find(';', p);
        const std::string args = code.substr(p, (stmt_end == std::string::npos ? code.size() : stmt_end) - p);
        if (args.find("sem::") == std::string::npos) {
            found.emplace_back(
                "Semaphore constructed from a raw id (no sem:: token): an undeclared participant "
                "the census cannot see -- declare a SemaphoreBinding and construct from sem::<name>");
        }
    }
    return found;
}

// Raw (undeclared) semaphore access patterns, each matched on its own to catch the two-statement
// form where the address is computed on a previous line. A declared binding never needs these:
// up()/down() call them from the framework header, not from kernel source.
std::vector<std::string> find_raw_semaphore_uses(const std::string& code) {
    static const char* kPatterns[] = {
        "get_semaphore(",                             // turning a semaphore id into a raw address
        "get_semaphore<",                             // same, templated spelling: get_semaphore<X>(id)
        "noc_semaphore_inc(",                         // atomic increment at a raw address (local or remote)
        "noc_semaphore_inc<",                         // same, templated spelling
        "noc_semaphore_inc_multicast<",               // templated multicast increment
        "noc_semaphore_set_multicast_loopback_src(",  // multicast set incl. the source core
        "noc_semaphore_set(",                         // raw set
        "noc_semaphore_set_remote(",                  // remote set
        "noc_semaphore_set_multicast(",               // multicast set
        "noc_semaphore_inc_multicast(",               // multicast increment
        "noc_semaphore_wait(",                        // raw wait: an undeclared READER still splits a
        "noc_semaphore_wait_min(",                    //   relocated cached semaphore (hangs forever)
        "noc_fast_atomic_increment(",                 // low-level NoC atomic (what noc_semaphore_inc wraps)
        "noc_fast_atomic_increment<",                 // same, templated spelling
        "noc_fast_atomic_cas4(",                      // the EXTERNAL down() lock primitive
        "noc_fast_atomic_cas4<",                      // same, templated spelling
        "MEM_NOC_CAS_RET_BASE",                       // reserved-region address arithmetic: stomping
        "MEM_NOC_SEM_LOCK_BASE",                      //   a lock word strands EXTERNAL down() forever;
        "MEM_DM_CACHED_SEM_BASE",                     //   a pool word is a live cached counter
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
// Entries are anchored on '/' so a file merely CONTAINING an entry's name is not exempt.
bool is_allowlisted(const std::string& path) {
    // Hardware keystone probes: deliberately issue raw NoC atomics at SCRATCH L1 addresses
    // (never at a declared semaphore) to measure hardware behaviour -- raw access is the thing
    // under test.
    if (path.find("test_kernels/dataflow/dm_cacheline_probe.cpp") != std::string::npos ||
        path.find("test_kernels/dataflow/noc_self_atomic.cpp") != std::string::npos ||
        path.find("test_kernels/dataflow/noc_atomic_ops_probe.cpp") != std::string::npos ||
        path.find("test_kernels/dataflow/noc_self_cas_drain.cpp") != std::string::npos) {
        return true;
    }
    // Residency instrumentation: reads its OWN declared semaphore's kernel_config ring slot
    // (read-only, uncached alias) to prove a cached count lives in the pool; no undeclared writer.
    if (path.find("test_kernels/dataflow/sem_census_probe.cpp") != std::string::npos) {
        return true;
    }
    // Legacy sdpa_decode helper header, classified only via includer propagation: the Metal 2.0
    // sampling writer includes it for tile-fill helpers and never instantiates the mcast function
    // that raw-constructs Semaphore<>(mcast_sem_id). A Metal 2.0 kernel that ever calls that
    // mcast path must declare a SemaphoreBinding instead.
    if (path.find("sdpa_decode/device/kernels/dataflow/dataflow_common.hpp") != std::string::npos) {
        return true;
    }
    // Watcher fault injection: issues a raw multicast atomic at an INVALID data-buffer range to
    // trip the watcher's sanitizer; no semaphore is involved, the malformed op is the thing
    // under test.
    if (path.find("test_kernels/dataflow/dram_copy_to_noc_coord_2_0.cpp") != std::string::npos) {
        return true;
    }
    return false;
}

// Kernel sources worth scanning: the metal test kernels and the TT-NN op kernels. Headers are
// included too -- a Metal 2.0 .cpp could otherwise hide a violation in an #include. A header is
// scanned when it self-classifies as Metal 2.0 OR when a Metal 2.0 file #includes it (includer
// propagation in the sweep below); legacy-only headers stay out.
bool is_kernel_source(const std::filesystem::path& p) {
    const auto ext = p.extension();
    if (ext != ".cpp" && ext != ".hpp" && ext != ".h") {
        return false;
    }
    // Any directory whose name ends in "kernels": kernels/, test_kernels/, <op>_kernels/.
    return p.string().find("kernels/") != std::string::npos;
}

// The paths a source #includes, from RAW text (quoted include paths must not be stripped as
// string literals). Full include strings are kept: resolution is path-aware (see the sweep)
// because bare-basename matching over-scans.
std::vector<std::string> include_paths(const std::string& raw_text) {
    std::vector<std::string> out;
    size_t pos = 0;
    while ((pos = raw_text.find("#include", pos)) != std::string::npos) {
        const size_t line_end = raw_text.find('\n', pos);
        std::string line = raw_text.substr(pos, (line_end == std::string::npos ? raw_text.size() : line_end) - pos);
        pos += 8;
        size_t open = line.find_first_of("\"<");
        if (open == std::string::npos) {
            continue;
        }
        const char close_ch = (line[open] == '<') ? '>' : '"';
        const size_t close = line.find(close_ch, open + 1);
        if (close == std::string::npos) {
            continue;
        }
        out.push_back(line.substr(open + 1, close - open - 1));
    }
    return out;
}

// POSITIVE CONTROL for the detector itself: if strip_comments or the pattern list silently
// stopped matching, the sweep below would still report zero violations and pass.
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
        // Address computed on a PREVIOUS line -- the two-statement form.
        {"split_noc_poke",
         "void kernel_main() {\n    uint64_t a = get_noc_addr(1, 1, 64);\n    noc_semaphore_inc(a, 1);\n}",
         true},
        {"raw_local_set", "void kernel_main() { noc_semaphore_set(ptr, 5); }", true},
        // Explicit <> pins the class-default scope and access rights instead of the baked ones.
        {"pinned_scope_ctor", "void kernel_main() { Semaphore<> s(sem::counter); }", true},
        // Brace-init and line-wrapped pins are the same violation in other spellings.
        {"pinned_scope_brace_init", "void kernel_main() { Semaphore<> s{sem::counter}; }", true},
        {"pinned_scope_line_wrapped", "void kernel_main() {\n    Semaphore<>\n        s(sem::counter);\n}", true},
        // A raw-id construction (any template spelling) is an undeclared census participant.
        {"raw_id_construction", "void kernel_main() { Semaphore<> s(sem_id); }", true},
        {"raw_id_construction_untemplated", "void kernel_main() { Semaphore s(get_arg_val<uint32_t>(0)); }", true},
        // ...but a scope-generic reference PARAMETER is not a construction and is fine.
        {"semaphore_ref_param_ok",
         "template <SemScope S> void helper(Semaphore<ProgrammableCoreType::TENSIX, S>& s) { s.up(1); }\n"
         "void kernel_main() { Semaphore s(sem::counter); helper(s); }",
         false},
        // Two-step token construction is flagged CONSERVATIVELY (the detector cannot prove `t`
        // is a token): construct from sem::<name> directly, or allowlist with a reason.
        {"two_step_token_construction_flagged", "void kernel_main() { auto t = sem::counter; Semaphore s(t); }", true},
        {"multicast_set", "void kernel_main() { noc_semaphore_set_multicast(a, b, 1); }", true},
        // Templated spelling of noc_semaphore_inc.
        {"templated_inc", "void kernel_main() { noc_semaphore_inc<true>(a, 1); }", true},
        // Templated spelling of get_semaphore.
        {"templated_get_semaphore",
         "void kernel_main() { uint32_t a = get_semaphore<ProgrammableCoreType::TENSIX>(0); }",
         true},
        // Each remaining kPatterns entry must trip the detector on its own.
        {"remote_set", "void kernel_main() { noc_semaphore_set_remote(a, b); }", true},
        {"multicast_loopback_src", "void kernel_main() { noc_semaphore_set_multicast_loopback_src(a, b, 1); }", true},
        {"multicast_inc", "void kernel_main() { noc_semaphore_inc_multicast(a, 1, 2); }", true},
        {"templated_multicast_inc", "void kernel_main() { noc_semaphore_inc_multicast<true>(a, 1, 2); }", true},
        // Raw wait: an undeclared READER splits a relocated cached semaphore.
        {"raw_wait", "void kernel_main() { noc_semaphore_wait(p, 1); }", true},
        {"raw_wait_min", "void kernel_main() { noc_semaphore_wait_min(p, 1); }", true},
        // Low-level NoC atomics and reserved-region address arithmetic.
        {"fast_atomic_inc", "void kernel_main() { noc_fast_atomic_increment(0, b, a, 4, 1, 31, 1, false); }", true},
        {"fast_atomic_cas", "void kernel_main() { noc_fast_atomic_cas4<0, true>(0, b, a, 4, 0, 1, false); }", true},
        {"lock_region_arith", "void kernel_main() { auto p = (uint32_t*)(MEM_NOC_SEM_LOCK_BASE + 4); }", true},
        {"cached_pool_arith", "void kernel_main() { auto p = (uint32_t*)(MEM_DM_CACHED_SEM_BASE); }", true},
        {"cas_ret_arith", "void kernel_main() { auto p = (uint32_t*)(MEM_NOC_CAS_RET_BASE + 8); }", true},
        // Patterns inside string literals are not code.
        {"string_literal_pattern", "void kernel_main() { const char* s = \"noc_semaphore_inc(\"; }", false},
        // A "/*" inside a string must not swallow the real statement after it.
        {"comment_start_in_string", "void kernel_main() { const char* s = \"/*\"; noc_semaphore_set(ptr, 5); }", true},
        // A digit separator is not a char-literal opener.
        {"digit_separator", "void kernel_main() { uint32_t n = 1'000; noc_semaphore_set(p, n); }", true},
        // A declared binding: the managed accessor only, no raw primitive in kernel source.
        {"clean_declared", "void kernel_main() {\n    Semaphore s(sem::counter);\n    s.up(1);\n}", false},
        // Prose mentioning the patterns must NOT be flagged (line and block comments).
        {"comment_only_line",
         "// calls get_semaphore(id) and noc_semaphore_inc(addr, 1)\nvoid kernel_main() {}",
         false},
        {"comment_only_block",
         "/*\n * uses get_semaphore(x)\n * and noc_semaphore_inc(y, 1)\n */\nvoid kernel_main() {}",
         false},
    };

    for (const auto& c : cases) {
        auto found = find_raw_semaphore_uses(strip_comments(c.body));
        for (auto& p : find_pinned_scope_constructions(strip_comments(c.body))) {
            found.push_back(p);
        }
        for (auto& p : find_unbound_semaphore_constructions(strip_comments(c.body))) {
            found.push_back(p);
        }
        if (c.should_flag) {
            EXPECT_FALSE(found.empty()) << "detector MISSED a violation (raw access or pinned scope) in case '"
                                        << c.name << "' -- the sweep test would silently pass on this code";
        } else {
            EXPECT_TRUE(found.empty()) << "detector FALSE-POSITIVED on case '" << c.name
                                       << "' (first: " << (found.empty() ? std::string{} : found.front()) << ")";
        }
    }
}

// The three FILE GATES the sweep routes every path through, plus detector discrimination:
// a regression in any of these leaves the sweep silently scanning the wrong set.
TEST(Metal2SemaphoreHygiene, FileGatesAndDetectorDiscrimination) {
    // is_kernel_source: all three kernel-directory spellings, and extension filtering.
    EXPECT_TRUE(is_kernel_source("tests/tt_metal/tt_metal/test_kernels/dataflow/foo.cpp"));
    // Synthetic paths: the predicate keys on extension + a *kernels/ component. The real op-tree
    // root is kept out of these literals (CI forbidden-imports word budget).
    EXPECT_TRUE(is_kernel_source("op_tree/operations/x/device/kernels/dataflow/foo.cpp"));
    EXPECT_TRUE(is_kernel_source("op_tree/operations/moreh/moreh_getitem_kernels/writer.cpp"));
    EXPECT_TRUE(is_kernel_source("op_tree/operations/x/device/kernels/helper.hpp"));
    EXPECT_FALSE(is_kernel_source("op_tree/operations/x/device/kernels/notes.md"));
    EXPECT_FALSE(is_kernel_source("op_tree/operations/x/device/microkernels_old/foo.cpp"));
    // is_allowlisted: directory-anchored -- a same-named file elsewhere is NOT exempt.
    EXPECT_TRUE(is_allowlisted("/repo/tests/tt_metal/tt_metal/test_kernels/dataflow/noc_self_atomic.cpp"));
    EXPECT_FALSE(is_allowlisted("/repo/other_op/device/kernels/noc_self_atomic.cpp"));
    // looks_like_metal2_kernel: all five generated namespaces classify.
    EXPECT_TRUE(looks_like_metal2_kernel("tensor::in0"));
    EXPECT_TRUE(looks_like_metal2_kernel("scratch::tmp"));
    EXPECT_FALSE(looks_like_metal2_kernel("noc_semaphore_inc(a, 1);"));  // raw-only legacy: skipped
    // include_paths: quoted and bracketed forms, full path preserved.
    const auto incs = include_paths("#include \"a/b/common.hpp\"\n#include <cstdint>\n#include \"local.h\"\n");
    ASSERT_EQ(incs.size(), 3u);
    EXPECT_EQ(incs[0], "a/b/common.hpp");
    EXPECT_EQ(incs[1], "cstdint");
    EXPECT_EQ(incs[2], "local.h");
    // Discrimination: the raw-id construction trips ONLY the unbound-ctor detector, and the
    // token construction trips NEITHER.
    EXPECT_TRUE(find_pinned_scope_constructions(strip_comments("Semaphore<> s(sem_id);")).empty());
    EXPECT_FALSE(find_unbound_semaphore_constructions(strip_comments("Semaphore<> s(sem_id);")).empty());
    EXPECT_TRUE(find_unbound_semaphore_constructions(strip_comments("Semaphore s(sem::counter);")).empty());
}

TEST(Metal2SemaphoreHygiene, NoRawSemaphoreAccessInMetal2Kernels) {
    // Locate the repo from __FILE__ so the lint cannot pass vacuously when TT_METAL_HOME is
    // unset; the env var is only a fallback for an out-of-tree layout.
    std::filesystem::path root;
    {
        const std::string self{__FILE__};
        const std::string marker{"/tests/tt_metal/"};
        // rfind: a checkout path that itself contains the marker must not truncate early.
        const size_t at = self.rfind(marker);
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
    // Broad roots: all of tests/ (the TT-NN test tree also holds Metal 2.0 kernels) plus the
    // whole op tree; is_kernel_source narrows per file.
    const std::vector<std::filesystem::path> scan_roots = {
        root / "tests",
        root / "ttnn",
    };

    std::vector<std::string> violations;
    uint32_t metal2_scanned = 0;
    uint32_t files_scanned = 0;

    // Retained kernel-dir headers, keyed by lexically-normal path, for includer propagation.
    // Resolution is PATH-AWARE (includer's directory first, then a path-suffix match); bare
    // basenames never match -- two unrelated ops can both own a "dataflow_common.hpp".
    struct HeaderInfo {
        std::string path;
        std::string code;  // stripped
        std::string raw;   // for include extraction
        bool classified = false;
    };
    std::map<std::string, HeaderInfo*> headers_by_path;
    std::vector<std::unique_ptr<HeaderInfo>> headers;
    // (includer path, include string) pairs from every classified file, resolved below.
    std::vector<std::pair<std::string, std::string>> pending_includes;

    auto scan_classified = [&](const std::string& path_str, const std::string& code) {
        metal2_scanned++;
        if (is_allowlisted(path_str)) {
            return;
        }
        for (const auto& pat : find_raw_semaphore_uses(code)) {
            violations.push_back(path_str + "  uses raw: " + pat);
        }
        for (const auto& pat : find_pinned_scope_constructions(code)) {
            violations.push_back(path_str + "  pins scope: " + pat);
        }
        for (const auto& pat : find_unbound_semaphore_constructions(code)) {
            violations.push_back(path_str + "  unbound ctor: " + pat);
        }
    };

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
            // Classify on stripped text: a legacy kernel whose COMMENTS mention sem:: must not be
            // treated as Metal 2.0 (its legitimate raw calls would then fail the suite).
            const std::string code = strip_comments(text);
            const std::string path_str = entry.path().string();
            const bool self_classified = looks_like_metal2_kernel(code);
            if (entry.path().extension() != ".cpp") {
                // Header: retain for propagation whether or not it self-classifies.
                headers.push_back(std::make_unique<HeaderInfo>(HeaderInfo{path_str, code, text, self_classified}));
                headers_by_path.emplace(entry.path().lexically_normal().string(), headers.back().get());
            }
            if (!self_classified) {
                continue;  // legacy (or marker-free helper): may still be classified via includers
            }
            scan_classified(path_str, code);
            for (auto& inc : include_paths(text)) {
                pending_includes.emplace_back(path_str, std::move(inc));
            }
        }
    }

    // Includer propagation to a fixed point: a retained header included from a classified file
    // becomes classified, is scanned, and its own includes propagate in turn. Resolution: the
    // includer's own directory first, then a '/'-anchored path-suffix match; unresolved includes
    // (system/api headers outside the kernel dirs) are skipped.
    while (!pending_includes.empty()) {
        const auto [includer, inc] = std::move(pending_includes.back());
        pending_includes.pop_back();
        std::vector<HeaderInfo*> hits;
        const std::string sibling = (std::filesystem::path(includer).parent_path() / inc).lexically_normal().string();
        if (auto it = headers_by_path.find(sibling); it != headers_by_path.end()) {
            hits.push_back(it->second);
        } else if (inc.find('/') != std::string::npos) {
            const std::string suffix = "/" + inc;
            for (const auto& [hpath, h] : headers_by_path) {
                if (hpath.size() > suffix.size() &&
                    hpath.compare(hpath.size() - suffix.size(), suffix.size(), suffix) == 0) {
                    hits.push_back(h);
                }
            }
        }
        for (HeaderInfo* h : hits) {
            if (h->classified) {
                continue;
            }
            h->classified = true;
            scan_classified(h->path, h->code);
            for (auto& next : include_paths(h->raw)) {
                pending_includes.emplace_back(h->path, std::move(next));
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
           "\n    Semaphore s(sem::<accessor_name>);   // CTAD adopts the baked scope and access rights"
           "\nIf the raw access is genuinely required (e.g. a hardware probe on scratch L1, not on a "
           "declared semaphore), add the file to is_allowlisted() with a reason."
           "\n\nOffending files:"
        << report;
}

}  // namespace
}  // namespace tt::tt_metal
