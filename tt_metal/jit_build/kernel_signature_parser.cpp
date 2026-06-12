// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "jit_build/kernel_signature_parser.hpp"

#include <cctype>
#include <stdexcept>
#include <string>
#include <string_view>
#include <vector>

#include <tracy/Tracy.hpp>

namespace tt::tt_metal {

namespace {

constexpr std::string_view kMarker = "TT_KERNEL";

bool is_ident_char(char c) { return std::isalnum(static_cast<unsigned char>(c)) || c == '_'; }

bool is_identifier(const std::string& s) {
    if (s.empty() || std::isdigit(static_cast<unsigned char>(s.front()))) {
        return false;
    }
    for (char c : s) {
        if (!is_ident_char(c)) {
            return false;
        }
    }
    return true;
}

bool is_ws(char c) { return std::isspace(static_cast<unsigned char>(c)) != 0; }

std::string trim(const std::string& s) {
    size_t b = 0;
    size_t e = s.size();
    while (b < e && is_ws(s[b])) {
        ++b;
    }
    while (e > b && is_ws(s[e - 1])) {
        --e;
    }
    return s.substr(b, e - b);
}

// Step 1: return a same-length copy of the source with every comment, string/char literal, and
// preprocessor-directive line blanked to spaces. This leaves the real code — identifiers and the
// structural punctuation < > ( ) { } , — at its original positions, with all the noise turned to
// whitespace. That single property is what lets the scan below find the lone TT_KERNEL token and
// bracket-match the template/parameter lists without ever matching a TT_KERNEL (or a stray
// bracket/comma) hiding inside a comment, a string, or the `#define TT_KERNEL ...` line.
//
// Limitation: raw string literals (R"(...)") are not lexed specially — the opening `"` starts an
// ordinary string literal, so an embedded unescaped `"` ends string-tracking early. Kernels don't
// use raw strings, so a TT_KERNEL hidden inside one is an accepted blind spot.
std::string strip_noise(const std::string& s) {
    std::string out(s.size(), ' ');
    enum class St { Normal, LineComment, BlockComment, String, Char, Preproc };
    St st = St::Normal;
    bool at_line_start = true;  // only whitespace seen since the last newline
    size_t i = 0;
    const size_t n = s.size();
    while (i < n) {
        const char c = s[i];
        const char nx = (i + 1 < n) ? s[i + 1] : '\0';
        switch (st) {
            case St::Normal:
                if (at_line_start && c == '#') {
                    st = St::Preproc;
                    ++i;
                } else if (c == '/' && nx == '/') {
                    st = St::LineComment;
                    i += 2;
                } else if (c == '/' && nx == '*') {
                    st = St::BlockComment;
                    i += 2;
                } else if (c == '"') {
                    st = St::String;
                    ++i;
                } else if (c == '\'') {
                    st = St::Char;
                    ++i;
                } else {
                    out[i] = c;
                    if (c == '\n') {
                        at_line_start = true;
                    } else if (!is_ws(c)) {
                        at_line_start = false;
                    }
                    ++i;
                }
                break;
            case St::LineComment:
                if (c == '\\' && nx == '\n') {
                    i += 2;  // line continuation: the // comment continues onto the next line
                } else if (c == '\n') {
                    out[i] = '\n';
                    at_line_start = true;
                    st = St::Normal;
                    ++i;
                } else {
                    ++i;
                }
                break;
            case St::BlockComment:
                if (c == '*' && nx == '/') {
                    st = St::Normal;
                    i += 2;
                } else {
                    if (c == '\n') {
                        out[i] = '\n';
                    }
                    ++i;
                }
                break;
            case St::String:
                if (c == '\\') {
                    i += 2;  // skip escaped char
                } else if (c == '"') {
                    st = St::Normal;
                    ++i;
                } else {
                    if (c == '\n') {
                        out[i] = '\n';
                    }
                    ++i;
                }
                break;
            case St::Char:
                if (c == '\\') {
                    i += 2;
                } else if (c == '\'') {
                    st = St::Normal;
                    ++i;
                } else {
                    ++i;
                }
                break;
            case St::Preproc:
                if (c == '\\' && nx == '\n') {
                    i += 2;  // line continuation
                } else if (c == '\n') {
                    out[i] = '\n';
                    at_line_start = true;
                    st = St::Normal;
                    ++i;
                } else {
                    ++i;
                }
                break;
        }
    }
    return out;
}

// Split a parameter-list body on top-level commas (commas not nested inside <> () [] {}).
std::vector<std::string> split_top_level_commas(const std::string& s) {
    std::vector<std::string> out;
    int depth = 0;
    size_t start = 0;
    for (size_t i = 0; i < s.size(); ++i) {
        const char c = s[i];
        if (c == '<' || c == '(' || c == '[' || c == '{') {
            ++depth;
        } else if (c == '>' || c == ')' || c == ']' || c == '}') {
            --depth;
        } else if (c == ',' && depth == 0) {
            out.push_back(s.substr(start, i - start));
            start = i + 1;
        }
    }
    out.push_back(s.substr(start));
    return out;
}

// Strip an optional leading or trailing `const` qualifier from a type spelling, so both
// `const uint32_t x` and `uint32_t const x` are accepted in Phase 1. Only `const` is handled — it
// is the one cv-qualifier a user is at all likely to write on a by-value kernel argument. A bare
// `const` with no base type is left intact so it still fails the uint32_t check below.
std::string strip_const(const std::string& type) {
    constexpr std::string_view kw = "const";
    std::string t = type;
    if (t.size() > kw.size() && std::string_view(t).substr(0, kw.size()) == kw && is_ws(t[kw.size()])) {
        t = trim(t.substr(kw.size()));
    }
    if (t.size() > kw.size() && std::string_view(t).substr(t.size() - kw.size()) == kw &&
        is_ws(t[t.size() - kw.size() - 1])) {
        t = trim(t.substr(0, t.size() - kw.size()));
    }
    return t;
}

// Parse one `uint32_t <name>` parameter entry; return its name. `kind` is used only for error
// messages ("template parameter" / "function parameter").
std::string extract_param_name(const std::string& raw_entry, const char* kind) {
    const std::string entry = trim(raw_entry);
    // The trailing run of identifier characters is the parameter name; everything before is the
    // type spelling. This is what makes the parser type-agnostic-but-strict in Phase 1.
    size_t b = entry.size();
    while (b > 0 && is_ident_char(entry[b - 1])) {
        --b;
    }
    const std::string name = entry.substr(b);
    const std::string type = strip_const(trim(entry.substr(0, b)));

    if (!is_identifier(name)) {
        throw std::runtime_error(
            std::string("TT_KERNEL ") + kind + " '" + entry +
            "' is missing a valid name (defaulted/unnamed/variadic parameters are not supported in Phase 1)");
    }
    if (type != "uint32_t" && type != "std::uint32_t") {
        throw std::runtime_error(
            std::string("TT_KERNEL ") + kind + " '" + name + "' has unsupported type '" + type +
            "' (only uint32_t is supported in Phase 1)");
    }
    return name;
}

std::vector<std::string> extract_param_names(const std::string& list_body, const char* kind) {
    std::vector<std::string> names;
    for (const std::string& piece : split_top_level_commas(list_body)) {
        if (trim(piece).empty()) {
            continue;  // empty list, or a trailing comma
        }
        names.push_back(extract_param_name(piece, kind));
    }
    return names;
}

// Read an identifier starting at or after `pos` (skipping leading whitespace). Advances `pos`
// past the identifier. Returns empty string if no identifier is found at that position.
std::string read_identifier(const std::string& s, size_t& pos) {
    while (pos < s.size() && is_ws(s[pos])) {
        ++pos;
    }
    const size_t start = pos;
    while (pos < s.size() && is_ident_char(s[pos])) {
        ++pos;
    }
    return s.substr(start, pos - start);
}

}  // namespace

std::optional<KernelMainSignature> parse_kernel_main_signature(const std::string& source) {
    // Profiler zone: measures the TT_KERNEL signature-parse overhead added to the JIT pipeline.
    // Captured under Tracy when TRACY_ENABLE is on; a no-op otherwise.
    ZoneScopedN("parse_kernel_main_signature");

    const std::string text = strip_noise(source);
    const size_t n = text.size();

    // Step 2: find the marker as a whole-word token. Zero -> not a TT_KERNEL kernel.
    std::vector<size_t> anchors;
    for (size_t p = text.find(kMarker); p != std::string::npos; p = text.find(kMarker, p + kMarker.size())) {
        const bool left_ok = (p == 0) || !is_ident_char(text[p - 1]);
        const size_t after = p + kMarker.size();
        const bool right_ok = (after >= n) || !is_ident_char(text[after]);
        if (left_ok && right_ok) {
            anchors.push_back(p);
        }
    }
    if (anchors.empty()) {
        return std::nullopt;
    }
    if (anchors.size() > 1) {
        throw std::runtime_error("found multiple TT_KERNEL markers; exactly one entry point per kernel is allowed");
    }
    const size_t anchor = anchors.front();

    KernelMainSignature sig;

    // Step 3: template head (look left). If the previous non-whitespace char is '>', there is a
    // template-parameter list; back-match the angle brackets and require a `template` keyword.
    {
        long k = static_cast<long>(anchor) - 1;
        while (k >= 0 && is_ws(text[k])) {
            --k;
        }
        if (k >= 0 && text[k] == '>') {
            int depth = 0;
            long m = k;
            for (; m >= 0; --m) {
                if (text[m] == '>') {
                    ++depth;
                } else if (text[m] == '<') {
                    if (--depth == 0) {
                        break;
                    }
                }
            }
            if (m < 0) {
                throw std::runtime_error("TT_KERNEL: unbalanced '<' '>' in the template parameter list");
            }
            const std::string tmpl_body = text.substr(m + 1, static_cast<size_t>(k) - (m + 1));

            // Require the `template` keyword immediately before the '<'.
            long t = m - 1;
            while (t >= 0 && is_ws(text[t])) {
                --t;
            }
            const std::string kw = "template";
            const long kw_start = t + 1 - static_cast<long>(kw.size());
            const bool kw_ok = kw_start >= 0 && text.compare(kw_start, kw.size(), kw) == 0 &&
                               (kw_start == 0 || !is_ident_char(text[kw_start - 1]));
            if (!kw_ok) {
                throw std::runtime_error("TT_KERNEL: expected a 'template <...>' clause before the marker");
            }
            sig.template_param_names = extract_param_names(tmpl_body, "template parameter");
        }
    }

    // Step 4: function head (look right): expect `void <name> ( ... )`.
    size_t pos = anchor + kMarker.size();
    const std::string ret_type = read_identifier(text, pos);
    if (ret_type != "void") {
        throw std::runtime_error(std::string("TT_KERNEL entry must return void (Phase 1); found '") + ret_type + "'");
    }
    sig.name = read_identifier(text, pos);
    if (sig.name.empty()) {
        throw std::runtime_error("TT_KERNEL: could not find the entry function name after the return type");
    }
    while (pos < n && is_ws(text[pos])) {
        ++pos;
    }
    if (pos >= n || text[pos] != '(') {
        throw std::runtime_error(
            std::string("TT_KERNEL entry '") + sig.name + "': expected '(' to begin the parameter list");
    }
    int depth = 0;
    size_t q = pos;
    for (; q < n; ++q) {
        if (text[q] == '(') {
            ++depth;
        } else if (text[q] == ')') {
            if (--depth == 0) {
                break;
            }
        }
    }
    if (q >= n) {
        throw std::runtime_error(
            std::string("TT_KERNEL entry '") + sig.name + "': unbalanced parentheses in the parameter list");
    }
    const std::string fn_body = text.substr(pos + 1, q - (pos + 1));

    // Steps 5-7: split each list on top-level commas and extract (uint32_t) names.
    sig.fn_param_names = extract_param_names(fn_body, "function parameter");

    return sig;
}

}  // namespace tt::tt_metal
