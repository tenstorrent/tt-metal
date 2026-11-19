// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <array>
#include <cstddef>
#include <optional>
#include <tuple>
#include <type_traits>
#include <utility>

#include "hostdevcommon/dprint_common.h"
#include "hostdev/device_print_common.h"
#include "hostdev/device_print_structures.h"
#include "waypoint.h"
#include "internal/hw_thread.h"
#include "internal/risc_attribs.h"
#include "internal/debug/dprint_buffer.h"

#if !defined(ENV_LLK_INFRA)
#include "risc_common.h"
#endif

#if defined(KERNEL_BUILD)
#include "dprint_tile.h"
#endif

#ifndef PROCESSOR_INDEX
#define PROCESSOR_INDEX_DEFINED 0
#define PROCESSOR_INDEX internal_::get_hw_thread_idx()
#else
#define PROCESSOR_INDEX_DEFINED 1
#endif

#if defined(KERNEL_BUILD) && !defined(ENV_LLK_INFRA)
#define DEVICE_PRINT_IS_KERNEL 1
#else
#define DEVICE_PRINT_IS_KERNEL 0
#endif

#define DEVICE_PRINT_STRINGS_SECTION_NAME ".device_print_strings"
#define DEVICE_PRINT_STRINGS_INFO_SECTION_NAME ".device_print_strings_info"

// Start of the .device_print_strings_info section, which represents list of DevicePrintStringInfo structures.
extern char __device_print_strings_info_start[];

// Wrapper for compile-time strings stored in the .device_print_strings ELF section.
// The host-side parser can resolve the pointer back to the string content via the ELF.
struct ct_string {
    const char* ptr;
};

// Stores a string literal in the .device_print_strings ELF section and returns a ct_string
// that DEVICE_PRINT can serialize. The host parser resolves the address to read the string.
#define CTSTR(literal)                                                                                              \
    ([]() -> ct_string {                                                                                            \
        static const char allocated_ct_string[] __attribute__((section(DEVICE_PRINT_STRINGS_SECTION_NAME), used)) = \
            literal;                                                                                                \
        return ct_string{allocated_ct_string};                                                                      \
    }())

struct bf4_t {
    union {
        struct {
            uint8_t exponent;
            uint8_t mantissa;
        };
        uint16_t val;
    };
    bf4_t(uint16_t val) : val(val) {}
    bf4_t(uint8_t exponent, uint8_t mantissa) : exponent(exponent), mantissa(mantissa) {}
};

struct bf8_t {
    union {
        struct {
            uint8_t exponent;
            uint8_t mantissa;
        };
        uint16_t val;
    };
    bf8_t(uint16_t val) : val(val) {}
    bf8_t(uint8_t exponent, uint8_t mantissa) : exponent(exponent), mantissa(mantissa) {}
};

struct bf16_t {
    uint16_t val;
    bf16_t(uint16_t val) : val(val) {}
};

template <uint16_t len>
struct dp_typed_array_t {
    uint32_t* ptr;
    uint16_t type;
    dp_typed_array_t(uint16_t type, uint32_t* ptr) : ptr(ptr), type(type) {}
};

// Prints T's fully-qualified name, resolved at compile time. Tag type: the name string is allocated
// in .device_print_strings by the serializer, so nothing is carried here.
template <typename T>
struct dp_type_name_t {};

struct dp_top_callstack_t {
    std::uintptr_t pc;
    std::uintptr_t ra;
    std::uintptr_t skip_frames;

    dp_top_callstack_t(std::uintptr_t pc, std::uintptr_t ra, std::uintptr_t skip_frames) {
        // We do a few adjustments to make host side parsing easier:
        //  - subtract the kernel offset from both PC and RA (tt-metal specific)
        //  - rewind RA by 1 byte so it points into the CALL, not the return address
        //  - leave 0xF...F untouched: it's the sentinel for an invalid/unknown address
        std::uint32_t kernel_offset = 0;

#if DEVICE_PRINT_IS_KERNEL
        // The kernel offset is the address the kernel runs from, computed as the firmware computes it.
#if defined(ARCH_WORMHOLE) && defined(COMPILE_FOR_NCRISC)
        // Wormhole's NCRISC runs its kernel from IRAM, where the firmware copies it (kernel_vma).
        kernel_offset = MEM_NCRISC_KERNEL_BASE;
#else
        const std::uint32_t launch_index = *GET_MAILBOX_ADDRESS_DEV(launch_msg_rd_ptr);
        const auto& config = GET_MAILBOX_ADDRESS_DEV(launch[launch_index])->kernel_config;
#if defined(COMPILE_FOR_DRISC) || (defined(ARCH_WORMHOLE) && defined(COMPILE_FOR_ERISC))
        // A DRAM-core kernel, and a kernel on Wormhole's active Ethernet core, loads at its link address, not into
        // the kernel config buffer, so its kernel_text_offset is the absolute load address.
        kernel_offset = config.kernel_text_offset[internal_::get_hw_thread_idx()];
#else
        // kernel_config_base[] is indexed by ProgrammableCoreType. PROGRAMMABLE_CORE_TYPE is set by
        // the HAL JIT defines from the build's HalProgrammableCoreType.
        kernel_offset = config.kernel_config_base[PROGRAMMABLE_CORE_TYPE] +
                        config.kernel_text_offset[internal_::get_hw_thread_idx()];
#endif
#endif
#endif

        this->pc = pc == UINTPTR_MAX ? pc : (pc - kernel_offset);
        this->ra = ra == UINTPTR_MAX ? ra : (ra - kernel_offset - 1);
        this->skip_frames = skip_frames;
    }

    __attribute__((always_inline)) static inline dp_top_callstack_t current(std::uintptr_t skip_frames) {
        std::uintptr_t pc;
        asm volatile("auipc %[pc], 0\n" : [pc] "=r"(pc));
        std::uintptr_t ra = reinterpret_cast<std::uintptr_t>(__builtin_return_address(0));

        // the +1 is to ignore the current frame
        return dp_top_callstack_t(pc, ra, skip_frames + 1);
    }
};

// A print that compiles to nothing is still one statement, `do { } while (0)`, like an enabled one, so that a missing
// `;` is an error with printing off too, and not only once printing is turned on.
#ifdef UCK_CHLKC_UNPACK
#define DEVICE_PRINT_UNPACK(format, ...) DEVICE_PRINT(format, ##__VA_ARGS__)
#else
#define DEVICE_PRINT_UNPACK(format, ...) \
    do {                                 \
    } while (0)
#endif

#ifdef UCK_CHLKC_MATH
#define DEVICE_PRINT_MATH(format, ...) DEVICE_PRINT(format, ##__VA_ARGS__)
#else
#define DEVICE_PRINT_MATH(format, ...) \
    do {                               \
    } while (0)
#endif

#ifdef UCK_CHLKC_PACK
#define DEVICE_PRINT_PACK(format, ...) DEVICE_PRINT(format, ##__VA_ARGS__)
#else
#define DEVICE_PRINT_PACK(format, ...) \
    do {                               \
    } while (0)
#endif

#if defined(COMPILE_FOR_BRISC) || defined(COMPILE_FOR_NCRISC) || defined(COMPILE_FOR_DM)
#define DEVICE_PRINT_DATA0(format, ...)          \
    do {                                         \
        if (noc_index == 0) {                    \
            DEVICE_PRINT(format, ##__VA_ARGS__); \
        }                                        \
    } while (0)
#define DEVICE_PRINT_DATA1(format, ...)          \
    do {                                         \
        if (noc_index == 1) {                    \
            DEVICE_PRINT(format, ##__VA_ARGS__); \
        }                                        \
    } while (0)
#else
#define DEVICE_PRINT_DATA0(format, ...) \
    do {                                \
    } while (0)
#define DEVICE_PRINT_DATA1(format, ...) \
    do {                                \
    } while (0)
#endif

#if defined(DEBUG_PRINT_ENABLED) && !defined(FORCE_DPRINT_OFF)
// DEVICE_PRINT is one statement, `do { } while (0)`. Its arguments are the arguments of one call of a generic lambda
// that takes them by value (bit-fields, packed fields and volatile objects are copied), so each is evaluated exactly
// once, in the caller's scope, before the print takes the lock. The lambda checks the format string and calls
// send_message(); both are always_inline.
#define DEVICE_PRINT(_device_print_format, ...)                                                                     \
    do {                                                                                                            \
        [](auto... args) __attribute__((always_inline)) {                                                           \
            /* Validate format string syntax */                                                                     \
            static_assert(                                                                                          \
                device_print_detail::checks::is_valid_format_string(_device_print_format),                          \
                "Invalid format string: unescaped '{' must be followed by '{', '}', or a digit");                   \
            /* Validate placeholder format */                                                                       \
            static_assert(                                                                                          \
                !device_print_detail::checks::has_mixed_placeholders(_device_print_format),                         \
                "Cannot mix indexed ({0}) and non-indexed ({}) placeholders in the same format string");            \
            /* For indexed placeholders, validate no index exceeds argument count */                                \
            static_assert(                                                                                          \
                !device_print_detail::checks::has_indexed_placeholders(_device_print_format) ||                     \
                    device_print_detail::checks::get_max_index(_device_print_format) <                              \
                        device_print_detail::helpers::count_arguments(args...),                                     \
                "Placeholder index exceeds number of arguments");                                                   \
            /* For indexed placeholders, validate all arguments are referenced */                                   \
            static_assert(                                                                                          \
                !device_print_detail::checks::has_indexed_placeholders(_device_print_format) ||                     \
                    device_print_detail::checks::all_arguments_referenced(_device_print_format, args...),           \
                "All arguments must be referenced when using indexed placeholders");                                \
            /* For non-indexed placeholders, count must match argument count */                                     \
            static_assert(                                                                                          \
                device_print_detail::checks::has_indexed_placeholders(_device_print_format) ||                      \
                    device_print_detail::checks::count_placeholders(_device_print_format) ==                        \
                        device_print_detail::helpers::count_arguments(args...),                                     \
                "Number of {} placeholders must match number of arguments");                                        \
            /* A local Tag struct carries format/file/line as static constexprs. Two struct definitions in distinct \
             * scopes are distinct types, so each DEVICE_PRINT call site gets its own register_string_info          \
             * instantiation. */                                                                                    \
            struct _DevicePrintStringInfoTag {                                                                      \
                static constexpr const auto& format() { return _device_print_format; }                              \
                static constexpr const auto& file() { return __FILE__; }                                            \
                static constexpr std::size_t line() { return __LINE__; }                                            \
            };                                                                                                      \
            device_print_detail::send_message<_DevicePrintStringInfoTag>(args...);                                  \
        }(__VA_ARGS__);                                                                                             \
    } while (0)

#define DEVICE_PRINT_INITIALIZE_LOCK() device_print_detail::locking::initialize_lock()
#define DEVICE_PRINT_KERNEL_FINISHED() device_print_detail::locking::update_kernel_finished()

// For a function that prints and that the compiler would copy into many callers: while printing is on, the compiler
// keeps exactly one copy of it (neither inlined nor cloned; clang has no noclone). While printing is off it is empty.
#if defined(__clang__)
#define DEVICE_PRINT_FORCE_NOINLINE __attribute__((noinline))
#else
#define DEVICE_PRINT_FORCE_NOINLINE __attribute__((noinline, noclone))
#endif

#if DEVICE_PRINT_DISPATCH_ENABLED
// Hook used by the lock spin and wait_for_space when DEVICE_PRINT itself runs on the dispatch_s
// kernel: that kernel is the only thing draining the per-core L1 print buffers, so a busy-wait
// for space (or for the lock that a RISC waiting for space holds) would deadlock unless dispatch_s
// drives its own DEVICE_PRINT dispatcher inside the wait.
// Defined in cq_dispatch_subordinate.cpp (where the dispatcher global lives). Declared at
// global namespace so call sites inside device_print_detail::locking::* qualify with ::.
void device_print_dispatcher_execute_hook();
#endif

// The library functions: static (a build without LTO gets only those its prints call; `unused` for the builds that
// warn about unused functions), one shared copy each (noclone), optimize("Os") at -O0 (a fifth of the size), and no
// loop splitting of print_message()'s wait loop at -O3. clang (clangd, tooling) knows neither noclone nor optimize.
#if defined(__clang__)
#define DEVICE_PRINT_LIBRARY_FUNCTION __attribute__((noinline, unused))
#define DEVICE_PRINT_NO_LOOP_SPLIT
#elif !defined(__OPTIMIZE__)
#define DEVICE_PRINT_LIBRARY_FUNCTION __attribute__((noinline, noclone, unused, optimize("Os")))
#define DEVICE_PRINT_NO_LOOP_SPLIT
#else
#define DEVICE_PRINT_LIBRARY_FUNCTION __attribute__((noinline, noclone, unused))
#define DEVICE_PRINT_NO_LOOP_SPLIT __attribute__((optimize("no-split-loops")))
#endif

// The code that stores an argument is inlined into the call site, so that each argument stays in a register (or is an
// immediate) until its store.
#define DEVICE_PRINT_ALWAYS_INLINE __attribute__((always_inline)) inline
// Hides a value from the optimizer; no instruction on RV32 (on RV64 it would cost a sign extension, so it is empty).
#if defined(__riscv_xlen) && __riscv_xlen == 64
#define DEVICE_PRINT_REGISTER_HINT(...)
#else
#define DEVICE_PRINT_REGISTER_HINT(...) asm("" : __VA_ARGS__)
#endif

namespace device_print_detail {

// Forward declarations needed by register_string_info and register_type_name_string.
namespace formatting {
template <std::size_t N, typename... Args>
constexpr auto update_format_string(const char (&format)[N]);
template <typename T>
constexpr auto get_type_name_string();
}

// Helper to copy a string literal into a std::array.
template <std::size_t N>
constexpr std::array<char, N> literal_to_array(const char (&s)[N]) {
    std::array<char, N> a{};
    for (std::size_t i = 0; i < N; ++i) {
        a[i] = s[i];
    }
    return a;
}

// Helper function to solve linkage problem. Marked `static` so the template — and therefore
// every instantiation and every static local inside it — has internal linkage. Without
// internal linkage, statics inside a DEVICE_PRINT expanded in a vague-linkage caller (inline
// header function, function template, in-class member, lambda-in-template) inherit COMDAT linkage;
// that conflicts with the named-section attribute and silently sends them to `.data` instead
// of `.device_print_strings_info`.
//
// We compute the format string inside this template so that Tag only references compile-time
// literals, as it's a local class and thus can't access enclosing variables.
template <typename Tag, typename... ArgsT>
static DEVICE_PRINT_ALWAYS_INLINE std::uintptr_t register_string_info() {
    static constexpr auto kFormat =
        device_print_detail::formatting::update_format_string<sizeof(Tag::format()), ArgsT...>(Tag::format())
            .to_array();
    static constexpr auto kFile = literal_to_array(Tag::file());  // constexpr init has to be in one statement
    static const auto allocated_string __attribute__((section(DEVICE_PRINT_STRINGS_SECTION_NAME), used)) = kFormat;
    static const auto allocated_file_string __attribute__((section(DEVICE_PRINT_STRINGS_SECTION_NAME), used)) = kFile;
    // 16-byte aligned: the low 4 bits of the address are free, and print_message() carries the payload size in them.
    static structures::DevicePrintStringInfo allocated_string_info
        __attribute__((section(DEVICE_PRINT_STRINGS_INFO_SECTION_NAME), used, aligned(16))) = {
            allocated_string.data(), allocated_file_string.data(), Tag::line()};
    return reinterpret_cast<std::uintptr_t>(&allocated_string_info);
}

// Allocates T's name in .device_print_strings and returns the pointer. `static` for the same
// internal-linkage reason as register_string_info.
template <typename T>
static DEVICE_PRINT_ALWAYS_INLINE const char* register_type_name_string() {
    static constexpr auto kTypeName = formatting::get_type_name_string<T>().to_array();
    static const auto allocated_type_name __attribute__((section(DEVICE_PRINT_STRINGS_SECTION_NAME), used)) = kTypeName;
    return allocated_type_name.data();
}

template <typename BufferType>
using device_print_buffer_ptr = volatile tt_l1_ptr BufferType*;

namespace helpers {

template <typename... Args>
constexpr std::size_t count_arguments(const Args&...) {
    return sizeof...(Args);
}

// Helper to check if a character is a digit
constexpr bool is_digit(char c) { return c >= '0' && c <= '9'; }

// Compile-time string class for building the result
template <std::size_t N>
struct static_string {
    char data[N + 1];
    std::size_t size;

    constexpr static_string() : data{}, size(0) {}

    constexpr void push_back(char c) {
        if (size < N) {
            data[size++] = c;
            data[size] = '\0';  // Ensure null termination
        }
    }

    constexpr void push_back_uint32(uint32_t value) {
        // Special case for zero
        if (value == 0) {
            push_back('0');
            return;
        }

        int digits = 0;
        uint64_t digit_reader = 1;

        while (digit_reader <= value) {
            digit_reader *= 10;
            digits++;
        }
        digit_reader /= 10;

        for (int i = 0; i < digits; ++i) {
            char digit_char = '0' + (value / digit_reader) % 10;
            push_back(digit_char);
            digit_reader /= 10;
        }
    }

    constexpr std::array<char, N + 1> to_array() const {
        std::array<char, N + 1> arr = {};
        for (std::size_t i = 0; i < size; ++i) {
            arr[i] = data[i];
        }
        arr[size] = '\0';
        return arr;
    }

    template <std::size_t M>
    constexpr bool check(const char (&expected)[M]) const {
        if (size != M - 1) {
            return false;
        }
        for (std::size_t i = 0; i < size; ++i) {
            if (data[i] != expected[i]) {
                return false;
            }
        }
        return true;
    }

    constexpr const char* c_str() const { return data; }
};

}  // namespace helpers

namespace parsing {

// Helper struct to return both parsed value and new position
struct IndexParseResult {
    uint32_t value;
    std::size_t new_pos;
};

// Helper to parse an integer from format string starting at position i
// Returns the parsed value and the new position after the digits
constexpr IndexParseResult parse_index(const char* format, std::size_t i, std::size_t format_len) {
    uint32_t value = 0;
    std::size_t pos = i;
    while (pos < format_len && helpers::is_digit(format[pos])) {
        value = value * 10 + (format[pos] - '0');
        ++pos;
    }
    return IndexParseResult{value, pos};
}

// Token types for format string parsing
enum class TokenType {
    Placeholder,         // {} or {N}
    EscapedOpenBrace,    // {{
    EscapedCloseBrace,   // }}
    InvalidPlaceholder,  // Invalid { sequence
    RegularChar          // Any other character
};

// Result of parsing a single token from format string
struct FormatToken {
    TokenType type = TokenType::RegularChar;
    std::size_t end_pos = 0;  // Position to continue parsing from (after this token)

    // Placeholder-specific data
    std::optional<uint32_t> index;    // The N in {N}, or -1 for {} or escape sequences
    std::optional<char> fill;         // <a character other than '{' or '}'>
    std::optional<char> align;        // '<' | '>' | '^'
    std::optional<char> sign;         // '+' | '-' | ' ']
    bool use_alternate_form = false;  // '#'
    bool use_zero_padding = false;    // '0'
    std::optional<uint32_t> width;
    std::optional<uint32_t> precision;
    bool use_current_locale = false;  // 'L'
    std::optional<char> format_type;  // 'a' | 'A' | 'b' | 'B' | 'c' | 'd' | 'e' | 'E' | 'f' | 'F' | 'g' | 'G' | 'o' |
                                      // 'p' | 's' | 'x' | 'X' | '?'

    constexpr bool is_indexed() const { return index.has_value(); }
};

// Parse a single token from the format string at position i
// This is the main tokenizer that handles all format string elements:
// - Placeholders: {} and {N}
// - Escape sequences: {{ and }}
// - Regular characters
template <std::size_t N>
constexpr FormatToken parse_format_token(const char (&format)[N], std::size_t i) {
    constexpr std::size_t format_len = N - 1;

    if (i >= format_len) {
        return FormatToken{TokenType::RegularChar, i + 1};
    }

    char c = format[i];

    // Check for escaped opening brace {{
    if (c == '{' && i + 1 < format_len && format[i + 1] == '{') {
        return FormatToken{TokenType::EscapedOpenBrace, i + 2};
    }

    // Check for escaped closing brace }}
    if (c == '}' && i + 1 < format_len && format[i + 1] == '}') {
        return FormatToken{TokenType::EscapedCloseBrace, i + 2};
    }

    // Check for placeholder
    if (c == '{') {
        i++;
        if (i >= format_len) {
            // '{' at end of string is invalid
            return FormatToken{TokenType::InvalidPlaceholder, i};
        }

        // We are trying to mimic fmtlib format specifiers here:
        // replacement_field ::= "{" [arg_id] [":" (format_spec | chrono_format_spec)] "}"
        // arg_id            ::= integer | identifier
        // integer           ::= digit+
        // digit             ::= "0"..."9"
        // identifier        ::= id_start id_continue*
        // id_start          ::= "a"..."z" | "A"..."Z" | "_"
        // id_continue       ::= id_start | digit
        // But we don't support using identifiers to reduce kernel size, only integers for arg_id.

        // Regarding format_spec:
        // format_spec ::= [[fill]align][sign]["#"]["0"][width]["." precision]["L"][type]
        // fill        ::= <a character other than '{' or '}'>
        // align       ::= "<" | ">" | "^"
        // sign        ::= "+" | "-" | " "
        // width       ::= integer | "{" [arg_id] "}"
        // precision   ::= integer | "{" [arg_id] "}"
        // type        ::= "a" | "A" | "b" | "B" | "c" | "d" | "e" | "E" | "f" | "F" |
        //                 "g" | "G" | "o" | "p" | "s" | "x" | "X" | "?"
        // We don't support using arg_id for width/precision.

        // Initially we mark parsed_token as invalid for simpler code and update it as Placeholder if valid
        FormatToken parsed_token{TokenType::InvalidPlaceholder};

        // arg_id parsing
        if (helpers::is_digit(format[i])) {
            IndexParseResult result = parse_index(format, i, format_len);
            i = result.new_pos;
            parsed_token.index = result.value;
        }

        // Check for format specifier
        if (i < format_len && format[i] == ':') {
            i++;  // Skip ':'

            // Parse fill and align
            if (i < format_len && (format[i] == '<' || format[i] == '>' || format[i] == '^')) {
                parsed_token.align = format[i];
                i++;
            } else if (
                i + 1 < format_len && format[i] != '{' && format[i] != '}' &&
                (format[i + 1] == '<' || format[i + 1] == '>' || format[i + 1] == '^')) {
                parsed_token.fill = format[i];
                parsed_token.align = format[i + 1];
                i += 2;
            }

            // Parse sign
            if (i < format_len && (format[i] == '+' || format[i] == '-' || format[i] == ' ')) {
                parsed_token.sign = format[i];
                i++;
            }

            // Parse alternate form
            if (i < format_len && format[i] == '#') {
                parsed_token.use_alternate_form = true;
                i++;
            }

            // Parse zero padding
            if (i < format_len && format[i] == '0') {
                parsed_token.use_zero_padding = true;
                i++;
            }

            // Parse width
            if (i < format_len && helpers::is_digit(format[i])) {
                IndexParseResult result = parse_index(format, i, format_len);
                i = result.new_pos;
                parsed_token.width = result.value;
            }

            // Parse precision
            if (i + 1 < format_len && format[i] == '.' && helpers::is_digit(format[i + 1])) {
                i++;  // Skip '.'
                IndexParseResult result = parse_index(format, i, format_len);
                i = result.new_pos;
                parsed_token.precision = result.value;
            }

            // Parse locale option
            if (i < format_len && format[i] == 'L') {
                parsed_token.use_current_locale = true;
                i++;
            }

            // Parse format type
            if (i < format_len &&
                (format[i] == 'a' || format[i] == 'A' || format[i] == 'b' || format[i] == 'B' || format[i] == 'c' ||
                 format[i] == 'd' || format[i] == 'e' || format[i] == 'E' || format[i] == 'f' || format[i] == 'F' ||
                 format[i] == 'g' || format[i] == 'G' || format[i] == 'o' || format[i] == 'p' || format[i] == 's' ||
                 format[i] == 'x' || format[i] == 'X' || format[i] == '?')) {
                parsed_token.format_type = format[i];
                i++;
            }
        }

        // Check for closing brace
        if (i < format_len && format[i] == '}') {
            parsed_token.type = TokenType::Placeholder;
            parsed_token.end_pos = i + 1;
            return parsed_token;
        }

        parsed_token.end_pos = i;
        return parsed_token;
    }

    // Regular character
    return FormatToken{TokenType::RegularChar, i + 1};
}

}  // namespace parsing

namespace checks {

// Helper to validate format string for invalid brace sequences
// Returns true if format string is valid, false if it contains errors
template <std::size_t N>
constexpr bool is_valid_format_string(const char (&format)[N]) {
    for (std::size_t i = 0; i < N - 1;) {
        parsing::FormatToken token = parsing::parse_format_token(format, i);

        // Check for invalid placeholder syntax
        if (token.type == parsing::TokenType::InvalidPlaceholder) {
            return false;
        }

        i = token.end_pos;
    }
    return true;
}

// Helper to check for mixed placeholder styles (both {} and {N})
// This should fail validation per fmtlib rules
template <std::size_t N>
constexpr bool has_mixed_placeholders(const char (&format)[N]) {
    bool found_indexed = false;
    bool found_unindexed = false;

    for (std::size_t i = 0; i < N - 1;) {
        parsing::FormatToken token = parsing::parse_format_token(format, i);
        if (token.type == parsing::TokenType::Placeholder) {
            if (token.is_indexed()) {
                found_indexed = true;
            } else {
                found_unindexed = true;
            }
            if (found_indexed && found_unindexed) {
                return true;
            }
        }
        i = token.end_pos;
    }
    return false;
}

// Helper to detect if format string uses indexed placeholders ({0}, {1}, etc.)
// Returns true if ANY placeholder has an index
template <std::size_t N>
constexpr bool has_indexed_placeholders(const char (&format)[N]) {
    for (std::size_t i = 0; i < N - 1;) {
        parsing::FormatToken token = parsing::parse_format_token(format, i);
        if (token.type == parsing::TokenType::Placeholder && token.is_indexed()) {
            return true;
        }
        i = token.end_pos;
    }
    return false;
}

// Helper to find the maximum index used in format string
template <std::size_t N>
constexpr uint32_t get_max_index(const char (&format)[N]) {
    uint32_t max_index = 0;
    for (std::size_t i = 0; i < N - 1;) {
        parsing::FormatToken token = parsing::parse_format_token(format, i);
        if (token.type == parsing::TokenType::Placeholder && token.is_indexed()) {
            max_index = std::max(max_index, token.index.value());
        }
        i = token.end_pos;
    }
    return max_index;
}

// Helper to validate that all arguments are referenced in indexed format
// Returns true if all argument indices from 0 to arg_count-1 are used at least once
template <std::size_t N, typename... Args>
constexpr bool all_arguments_referenced(const char (&format)[N], const Args&... args) {
    if (sizeof...(args) == 0) {
        return true;
    }

    // Track which arguments are referenced
    std::array<bool, sizeof...(Args)> referenced = {};

    for (std::size_t i = 0; i < N - 1;) {
        parsing::FormatToken token = parsing::parse_format_token(format, i);
        if (token.type == parsing::TokenType::Placeholder && token.is_indexed()) {
            if (static_cast<std::size_t>(token.index.value()) < sizeof...(args)) {
                referenced[token.index.value()] = true;
            }
        }
        i = token.end_pos;
    }

    // Check that all arguments from 0 to arg_count-1 are referenced
    for (std::size_t i = 0; i < sizeof...(args); ++i) {
        if (!referenced[i]) {
            return false;
        }
    }
    return true;
}

// Helper to count placeholders in format string at compile time
template <std::size_t N>
constexpr std::size_t count_placeholders(const char (&format)[N]) {
    std::size_t count = 0;
    for (std::size_t i = 0; i < N - 1;) {
        parsing::FormatToken token = parsing::parse_format_token(format, i);
        if (token.type == parsing::TokenType::Placeholder) {
            ++count;
        }
        i = token.end_pos;
    }
    return count;
}

}  // namespace checks

namespace formatting {

struct device_print_type_info {
    char type_char;
    uint32_t size_in_bytes;
};

// A call site writes its payload into an image on its stack before print_message() takes the lock, or into the print
// buffer after print_message() reserved the message (see send_message): one store per argument (two for 8 bytes)
// through a volatile pointer, which GCC neither merges nor turns into constants from .rodata; may_alias types, as the
// library reads an image as words.
typedef uint16_t __attribute__((may_alias)) image_u16_t;
typedef uint32_t __attribute__((may_alias)) image_word_t;
typedef uint64_t __attribute__((may_alias)) image_u64_t;

// The destinations, as the types of their byte, halfword and word pointers.
struct to_image {
    using byte_ptr = volatile uint8_t*;
    using half_ptr = volatile image_u16_t*;
    using word_ptr = volatile image_word_t*;
    static constexpr bool aligned_8 = false;
};
// A 16-byte-aligned image that holds the 8-byte values first: they are 8-byte aligned in it.
struct to_store_image : to_image {
    static constexpr bool aligned_8 = true;
};
struct to_buffer {
    using byte_ptr = volatile tt_l1_ptr uint8_t*;
    using half_ptr = volatile tt_l1_ptr uint16_t*;
    using word_ptr = volatile tt_l1_ptr uint32_t*;
    static constexpr bool aligned_8 = false;
};

template <typename To>
DEVICE_PRINT_ALWAYS_INLINE void store_8bit_value(typename To::byte_ptr dst, uint32_t offset, uint8_t value) {
    dst[offset] = value;
}
template <typename To>
DEVICE_PRINT_ALWAYS_INLINE void store_16bit_value(typename To::byte_ptr dst, uint32_t offset, uint16_t value) {
    *reinterpret_cast<typename To::half_ptr>(dst + offset) = value;
}
template <typename To>
DEVICE_PRINT_ALWAYS_INLINE void store_32bit_value(typename To::byte_ptr dst, uint32_t offset, uint32_t value) {
    *reinterpret_cast<typename To::word_ptr>(dst + offset) = value;
}
// Store an 8-byte value as two 32-bit stores: arguments are only 4-byte aligned in a message (riscs with 8-byte
// registers fault on misaligned 8-byte stores). Into an 8-byte-aligned slot RV64 stores a value computed at run time
// with one store, and a constant as two words, which GCC builds in registers (for one store it would load the constant
// from .rodata); __builtin_constant_p() is folded once the print is inlined.
template <typename To>
DEVICE_PRINT_ALWAYS_INLINE void store_64bit_value(typename To::byte_ptr dst, uint32_t offset, uint64_t value) {
    if constexpr (To::aligned_8 && sizeof(std::uintptr_t) == 8) {
        if (!__builtin_constant_p(value)) {
            *reinterpret_cast<volatile image_u64_t*>(dst + offset) = value;
            return;
        }
    }
    auto words = reinterpret_cast<typename To::word_ptr>(dst + offset);
    words[0] = static_cast<uint32_t>(value);
    words[1] = static_cast<uint32_t>(value >> 32);
}

template <typename To>
DEVICE_PRINT_ALWAYS_INLINE void store_pointer_value(typename To::byte_ptr dst, uint32_t offset, std::uintptr_t value) {
    if constexpr (sizeof(std::uintptr_t) == 8) {
        store_64bit_value<To>(dst, offset, value);
    } else {
        store_32bit_value<To>(dst, offset, value);
    }
}

// Type-to-info mapping for format strings and serialization
template <typename T, typename = void>
struct device_print_type {
    static constexpr device_print_type_info value = {'#', 0};  // Unknown type default
    template <typename To>
    static void serialize(typename To::byte_ptr dst, uint32_t offset, T argument) {
        static_assert(!std::is_same_v<T, T>, "No serialization defined for this type");
    }
};

// Specializations for different types
template <>
struct device_print_type<std::int8_t> {
    static constexpr device_print_type_info value = {'b', sizeof(std::int8_t)};
    template <typename To>
    static void serialize(typename To::byte_ptr dst, uint32_t offset, int8_t argument) {
        store_8bit_value<To>(dst, offset, static_cast<uint8_t>(argument));
    }
};
template <>
struct device_print_type<std::uint8_t> {
    static constexpr device_print_type_info value = {'B', sizeof(std::uint8_t)};
    template <typename To>
    static void serialize(typename To::byte_ptr dst, uint32_t offset, uint8_t argument) {
        store_8bit_value<To>(dst, offset, argument);
    }
};
template <>
struct device_print_type<std::int16_t> {
    static constexpr device_print_type_info value = {'h', sizeof(std::int16_t)};
    template <typename To>
    static void serialize(typename To::byte_ptr dst, uint32_t offset, int16_t argument) {
        store_16bit_value<To>(dst, offset, static_cast<uint16_t>(argument));
    }
};
template <>
struct device_print_type<std::uint16_t> {
    static constexpr device_print_type_info value = {'H', sizeof(std::uint16_t)};
    template <typename To>
    static void serialize(typename To::byte_ptr dst, uint32_t offset, uint16_t argument) {
        store_16bit_value<To>(dst, offset, argument);
    }
};
template <>
struct device_print_type<std::int32_t> {
    static constexpr device_print_type_info value = {'i', sizeof(std::int32_t)};
    template <typename To>
    static void serialize(typename To::byte_ptr dst, uint32_t offset, int32_t argument) {
        store_32bit_value<To>(dst, offset, static_cast<uint32_t>(argument));
    }
};
template <>
struct device_print_type<std::uint32_t> {
    static constexpr device_print_type_info value = {'I', sizeof(std::uint32_t)};
    template <typename To>
    static void serialize(typename To::byte_ptr dst, uint32_t offset, uint32_t argument) {
        store_32bit_value<To>(dst, offset, argument);
    }
};
template <>
struct device_print_type<std::int64_t> {
    static constexpr device_print_type_info value = {'q', sizeof(std::int64_t)};
    template <typename To>
    static void serialize(typename To::byte_ptr dst, uint32_t offset, int64_t argument) {
        store_64bit_value<To>(dst, offset, static_cast<uint64_t>(argument));
    }
};
template <>
struct device_print_type<std::uint64_t> {
    static constexpr device_print_type_info value = {'Q', sizeof(std::uint64_t)};
    template <typename To>
    static void serialize(typename To::byte_ptr dst, uint32_t offset, uint64_t argument) {
        store_64bit_value<To>(dst, offset, argument);
    }
};
template <>
struct device_print_type<float> {
    static constexpr device_print_type_info value = {'f', sizeof(float)};
    template <typename To>
    static void serialize(typename To::byte_ptr dst, uint32_t offset, float argument) {
        store_32bit_value<To>(dst, offset, __builtin_bit_cast(uint32_t, argument));
    }
};
template <>
struct device_print_type<double> {
    static constexpr device_print_type_info value = {'d', sizeof(double)};
    template <typename To>
    static void serialize(typename To::byte_ptr dst, uint32_t offset, double argument) {
        store_64bit_value<To>(dst, offset, __builtin_bit_cast(uint64_t, argument));
    }
};
template <>
struct device_print_type<bool> {
    static constexpr device_print_type_info value = {'?', 1};
    template <typename To>
    static void serialize(typename To::byte_ptr dst, uint32_t offset, bool argument) {
        store_8bit_value<To>(dst, offset, static_cast<uint8_t>(argument));
    }
};
template <>
struct device_print_type<bf4_t> {
    static constexpr device_print_type_info value = {'e', sizeof(uint16_t)};
    static_assert(sizeof(bf4_t) == sizeof(uint16_t), "bf4_t must be 16 bits");
    template <typename To>
    static void serialize(typename To::byte_ptr dst, uint32_t offset, bf4_t argument) {
        store_16bit_value<To>(dst, offset, static_cast<uint16_t>(argument.val));
    }
};
template <>
struct device_print_type<bf8_t> {
    static constexpr device_print_type_info value = {'E', sizeof(uint16_t)};
    static_assert(sizeof(bf8_t) == sizeof(uint16_t), "bf8_t must be 16 bits");
    template <typename To>
    static void serialize(typename To::byte_ptr dst, uint32_t offset, bf8_t argument) {
        store_16bit_value<To>(dst, offset, static_cast<uint16_t>(argument.val));
    }
};
template <>
struct device_print_type<bf16_t> {
    static constexpr device_print_type_info value = {'w', sizeof(uint16_t)};
    static_assert(sizeof(bf16_t) == sizeof(uint16_t), "bf16_t must be 16 bits");
    template <typename To>
    static void serialize(typename To::byte_ptr dst, uint32_t offset, bf16_t argument) {
        store_16bit_value<To>(dst, offset, static_cast<uint16_t>(argument.val));
    }
};

// Catch-all for integral types that don't have an explicit specialization above (e.g. 'int' and
// 'unsigned int' on platforms where they differ from std::int32_t / std::uint32_t).
// Delegates to the matching fixed-width specialization based on size and signedness.
template <typename T>
struct device_print_type<
    T,
    std::enable_if_t<
        std::is_integral_v<T> && !std::is_same_v<T, bool> && !std::is_same_v<T, std::int8_t> &&
        !std::is_same_v<T, std::uint8_t> && !std::is_same_v<T, std::int16_t> && !std::is_same_v<T, std::uint16_t> &&
        !std::is_same_v<T, std::int32_t> && !std::is_same_v<T, std::uint32_t> && !std::is_same_v<T, std::int64_t> &&
        !std::is_same_v<T, std::uint64_t>>> {
    // Pick the fixed-width type with matching size and signedness
    using fixed_type = std::conditional_t<
        std::is_signed_v<T>,
        std::conditional_t<
            sizeof(T) == 1,
            std::int8_t,
            std::conditional_t<
                sizeof(T) == 2,
                std::int16_t,
                std::conditional_t<sizeof(T) == 4, std::int32_t, std::int64_t>>>,
        std::conditional_t<
            sizeof(T) == 1,
            std::uint8_t,
            std::conditional_t<
                sizeof(T) == 2,
                std::uint16_t,
                std::conditional_t<sizeof(T) == 4, std::uint32_t, std::uint64_t>>>>;

    static constexpr device_print_type_info value = device_print_type<fixed_type>::value;
    template <typename To>
    static void serialize(typename To::byte_ptr dst, uint32_t offset, T argument) {
        device_print_type<fixed_type>::template serialize<To>(dst, offset, static_cast<fixed_type>(argument));
    }
};

// Pointer types (including strings)
template <typename T>
struct device_print_type<T*> {
    static constexpr device_print_type_info value = {'p', sizeof(T*)};
    static_assert(sizeof(T*) == 4 || sizeof(T*) == 8, "Unsupported pointer size");
    template <typename To>
    static void serialize(typename To::byte_ptr dst, uint32_t offset, T* argument) {
        store_pointer_value<To>(dst, offset, reinterpret_cast<std::uintptr_t>(argument));
    }
};
template <>
struct device_print_type<char*> {
    static constexpr device_print_type_info value = {'s', sizeof(char*)};
    template <typename To>
    static void serialize(typename To::byte_ptr dst, uint32_t offset, char* argument) {
        store_pointer_value<To>(dst, offset, reinterpret_cast<std::uintptr_t>(argument));
    }
};
template <>
struct device_print_type<const char*> {
    static constexpr device_print_type_info value = {'s', sizeof(const char*)};
    template <typename To>
    static void serialize(typename To::byte_ptr dst, uint32_t offset, const char* argument) {
        store_pointer_value<To>(dst, offset, reinterpret_cast<std::uintptr_t>(argument));
    }
};

// Compile-time strings (CTSTR) — serialized as pointer, same type char 's' as const char*.
// The host parser distinguishes by checking if the address falls within .device_print_strings.
template <>
struct device_print_type<ct_string> {
    static constexpr device_print_type_info value = {'s', sizeof(const char*)};
    template <typename To>
    static void serialize(typename To::byte_ptr dst, uint32_t offset, ct_string argument) {
        store_pointer_value<To>(dst, offset, reinterpret_cast<std::uintptr_t>(argument.ptr));
    }
};

// Array types (treat as strings)
template <std::size_t N>
struct device_print_type<char[N]> {
    static constexpr device_print_type_info value = {'s', sizeof(const char*)};
};
template <std::size_t N>
struct device_print_type<const char[N]> {
    static constexpr device_print_type_info value = {'s', sizeof(const char*)};
};

// A lone TileSlice is its own image (send_message); next to other arguments it is copied word by word.
template <typename T>
inline constexpr bool is_tile_slice = false;
// Defined for the TileSlice types of kernel builds.
template <typename T>
const image_word_t* tile_slice_image(T& argument);

#if defined(KERNEL_BUILD)
template <int MAX_BYTES>
inline constexpr bool is_tile_slice<TileSlice<MAX_BYTES>> = true;

// The words of a TileSlice, read through a may_alias type (a TileSlice is not an array of uint32_t). The print's copy
// of a TileSlice is a local structure, which GCC word-aligns on RISC-V (LOCAL_ALIGNMENT), although TileSlice is packed.
typedef uint32_t __attribute__((may_alias)) tile_slice_word_t;

// Stores the size of the data of a print's own copy of a TileSlice in its `pad` byte and returns the copy as an image.
template <int MAX_BYTES>
DEVICE_PRINT_ALWAYS_INLINE const image_word_t* tile_slice_image(TileSlice<MAX_BYTES>& argument) {
    static_assert(
        sizeof(TileSlice<MAX_BYTES>) % 4 == 0, "TileSlice size must be a multiple of 4 bytes for proper serialization");
    static_assert(MAX_BYTES < 256, "MAX_BYTES must be less than 256 to fit size in a single byte");
    argument.pad = MAX_BYTES;  // The actual size
    return reinterpret_cast<const image_word_t*>(&argument);
}

// Copies a TileSlice word by word, then stores the size of its data in its `pad` byte.
template <typename To, uint32_t MAX_BYTES>
static DEVICE_PRINT_ALWAYS_INLINE void serialize_tile_slice(
    typename To::byte_ptr dst, uint32_t offset, const TileSlice<MAX_BYTES>& argument) {
    static_assert(
        sizeof(TileSlice<MAX_BYTES>) % 4 == 0, "TileSlice size must be a multiple of 4 bytes for proper serialization");
    static_assert(MAX_BYTES < 256, "MAX_BYTES must be less than 256 to fit size in a single byte");
    const tile_slice_word_t* argument_as_uint32_ptr = reinterpret_cast<const tile_slice_word_t*>(&argument);
#pragma GCC unroll 1
    for (uint32_t i = 0; i < sizeof(TileSlice<MAX_BYTES>); i += 4) {
        store_32bit_value<To>(dst, offset + i, argument_as_uint32_ptr[i / 4]);
    }
    store_8bit_value<To>(dst, offset + offsetof(TileSliceHostDev<MAX_BYTES>, pad), MAX_BYTES);  // The actual size
}

// TileSlice types
template <uint32_t MAX_BYTES>
struct device_print_type<TileSlice<MAX_BYTES>> {
    static constexpr device_print_type_info value = {'t', sizeof(TileSlice<MAX_BYTES>)};
    template <typename To>
    static void serialize(typename To::byte_ptr dst, uint32_t offset, const TileSlice<MAX_BYTES>& argument) {
        serialize_tile_slice<To, MAX_BYTES>(dst, offset, argument);
    }
};
template <>
struct device_print_type<TileSlice<64>> {
    static constexpr device_print_type_info value = {'t', sizeof(TileSlice<64>)};
    template <typename To>
    static void serialize(typename To::byte_ptr dst, uint32_t offset, const TileSlice<64>& argument) {
        serialize_tile_slice<To, 64>(dst, offset, argument);
    }
};
template <>
struct device_print_type<TileSlice<128>> {
    static constexpr device_print_type_info value = {'t', sizeof(TileSlice<128>)};
    template <typename To>
    static void serialize(typename To::byte_ptr dst, uint32_t offset, const TileSlice<128>& argument) {
        serialize_tile_slice<To, 128>(dst, offset, argument);
    }
};
#endif  // defined(KERNEL_BUILD)

// Copies the len elements of a typed array into the print buffer, under the lock (the typed arrays that
// device_print_typed_array() doesn't copy, see send_message). Call sites store typed arrays after the other arguments
// (see fill_payload), so that no other value has to be kept across the call.
static DEVICE_PRINT_LIBRARY_FUNCTION void serialize_typed_array(
    volatile tt_l1_ptr uint32_t* dst, const uint32_t* src, uint32_t len) {
#pragma GCC unroll 1
    for (uint32_t i = 0; i < len; ++i) {
        dst[i] = src[i];
    }
}

template <typename T>
inline constexpr bool is_typed_array = false;
// A typed array without elements is one word, (0 << 16) | type: a scalar (see word_argument).
template <uint16_t len>
inline constexpr bool is_typed_array<dp_typed_array_t<len>> = (len != 0);
template <typename T>
inline constexpr uint32_t typed_array_length = 0;
template <uint16_t len>
inline constexpr uint32_t typed_array_length<dp_typed_array_t<len>> = len;

// dp_typed_array_t: serialized as (len + 1) uint32_t elements: [(len << 16) | type, data[0..len-1]]
template <uint16_t len>
struct device_print_type<dp_typed_array_t<len>> {
    static constexpr device_print_type_info value = {'A', (len + 1) * sizeof(uint32_t)};
    // The first word, (len << 16) | type.
    DEVICE_PRINT_ALWAYS_INLINE static uint32_t first_word(const dp_typed_array_t<len>& argument) {
        return (static_cast<uint32_t>(len) << 16) | argument.type;
    }
    template <typename To>
    DEVICE_PRINT_ALWAYS_INLINE static void serialize(
        typename To::byte_ptr dst, uint32_t offset, dp_typed_array_t<len> argument) {
        store_32bit_value<To>(dst, offset, first_word(argument));
        if constexpr (len != 0 && len <= 2) {
            // A short array is copied here, as the elements' loads and stores cost no more than a call.
            for (uint32_t i = 0; i < len; ++i) {
                store_32bit_value<To>(dst, offset + 4 * (i + 1), argument.ptr[i]);
            }
        } else if constexpr (len != 0) {
            static_assert(
                std::is_same_v<To, to_buffer>, "a typed array is copied into the print buffer, never an image");
            serialize_typed_array(reinterpret_cast<volatile tt_l1_ptr uint32_t*>(dst + offset) + 1, argument.ptr, len);
        }
    }
};

template <typename T>
struct device_print_type<dp_type_name_t<T>> {
    static constexpr device_print_type_info value = device_print_type<ct_string>::value;
    template <typename To>
    static void serialize(typename To::byte_ptr dst, uint32_t offset, dp_type_name_t<T>) {
        device_print_type<ct_string>::template serialize<To>(dst, offset, ct_string{register_type_name_string<T>()});
    }
};

template <>
struct device_print_type<dp_top_callstack_t> {
    static_assert(
        sizeof(uintptr_t) == sizeof(uint32_t) || sizeof(uintptr_t) == sizeof(uint64_t), "Unsupported pointer size");

    static constexpr device_print_type_info value = {'c', 3 * sizeof(uintptr_t)};

    template <typename To>
    static void serialize(typename To::byte_ptr dst, uint32_t offset, dp_top_callstack_t argument) {
        store_pointer_value<To>(dst, offset, argument.pc);
        store_pointer_value<To>(dst, offset + sizeof(uintptr_t), argument.ra);
        store_pointer_value<To>(dst, offset + 2 * sizeof(uintptr_t), argument.skip_frames);
    }
};

// Enum types: serialized as their underlying type.
template <typename T>
struct device_print_type<T, std::enable_if_t<std::is_enum_v<T>>> {
    using underlying_type = std::underlying_type_t<T>;

    static constexpr device_print_type_info value = device_print_type<underlying_type>::value;

    template <typename To>
    static void serialize(typename To::byte_ptr dst, uint32_t offset, T argument) {
        device_print_type<underlying_type>::template serialize<To>(dst, offset, static_cast<underlying_type>(argument));
    }
};

// Helper to get type character for a single type, removing cv-qualifiers and references
template <typename T>
constexpr device_print_type_info get_type_info() {
    using base_type = std::remove_cv_t<std::remove_reference_t<T>>;
    return device_print_type<base_type>::value;
}

template <typename... Args>
constexpr std::array<device_print_type_info, sizeof...(Args)> get_types_info() {
    return {get_type_info<Args>()...};
}

template <typename... Args>
constexpr std::array<uint32_t, sizeof...(Args)> get_arg_reorder() {
    constexpr auto type_infos = get_types_info<Args...>();

    // Initialize to default ordering of arguments (0, 1, 2, ...).
    std::array<uint32_t, sizeof...(Args)> arg_reorder = {};
    for (std::size_t i = 0; i < arg_reorder.size(); ++i) {
        arg_reorder[i] = i;
    }

    // Sort arguments by size_in_bytes descending to optimize serialization (largest first)
    for (std::size_t i = 0; i < arg_reorder.size(); ++i) {
        for (std::size_t j = i + 1; j < arg_reorder.size(); ++j) {
            if (type_infos[arg_reorder[j]].size_in_bytes > type_infos[arg_reorder[i]].size_in_bytes) {
                uint32_t temp = arg_reorder[i];
                arg_reorder[i] = arg_reorder[j];
                arg_reorder[j] = temp;
            }
        }
    }
    return arg_reorder;
}

template <typename... Args>
constexpr std::array<uint32_t, sizeof...(Args)> get_arg_offsets() {
    constexpr auto type_infos = get_types_info<Args...>();
    constexpr auto arg_reorder = get_arg_reorder<Args...>();
    std::array<uint32_t, sizeof...(Args)> arg_memory_offsets = {};
    uint32_t current_offset = sizeof(structures::DevicePrintHeader::value);
    for (std::size_t i = 0; i < arg_memory_offsets.size(); ++i) {
        arg_memory_offsets[i] = current_offset;
        current_offset += type_infos[arg_reorder[i]].size_in_bytes;
    }
    std::array<uint32_t, sizeof...(Args)> arg_offset = {};
    for (std::size_t i = 0; i < arg_offset.size(); ++i) {
        arg_offset[arg_reorder[i]] = arg_memory_offsets[i];
    }
    return arg_offset;
}

// Extracts the fully-qualified type name from __PRETTY_FUNCTION__ at compile time. The result is
// sized exactly to the name, so to_array() yields a compact copy.
template <typename T>
constexpr auto get_type_name_string() {
#if !defined(__GNUC__)
    static_assert(false, "get_type_name_string requires __PRETTY_FUNCTION__ (GCC/Clang)");
#endif
    // Locates the type name in a __PRETTY_FUNCTION__ of the form "... [with T = test::deep::Enum1]",
    // returning it as {start, length}.
    constexpr auto find_type_name = [](const char* fn) {
        std::size_t fn_len = 0;
        while (fn[fn_len] != '\0') {
            fn_len++;
        }

        // Search for "T = " in "[with T = TypeName]"
        std::size_t start = 0;
        for (std::size_t i = 0; i + 3 < fn_len; i++) {
            if (fn[i] == 'T' && fn[i + 1] == ' ' && fn[i + 2] == '=' && fn[i + 3] == ' ') {
                start = i + 4;
                break;
            }
        }

        // The name ends at the first ';' — GCC appends "; other = ..." when the signature names further
        // template params or typedefs — or, when there is none, at the trailing ']'.
        std::size_t end = start;
        while (end < fn_len && fn[end] != ';') {
            end++;
        }
        return std::tuple<std::size_t, std::size_t>{start, (end == fn_len ? fn_len - 1 : end) - start};
    };

    constexpr auto type_name = find_type_name(__PRETTY_FUNCTION__);
    constexpr std::size_t start = std::get<0>(type_name);
    constexpr std::size_t length = std::get<1>(type_name);

    helpers::static_string<length> result;
    for (std::size_t i = 0; i < length; i++) {
        result.push_back(__PRETTY_FUNCTION__[start + i]);
    }
    return result;
}

// Returns the extra characters needed in the format string for an enum type argument.
// Non-enum types contribute 0 extra chars.
template <typename T>
constexpr std::size_t enum_extra_format_chars() {
    using base = std::remove_cv_t<std::remove_reference_t<T>>;
    if constexpr (std::is_enum_v<base>) {
        constexpr auto name = get_type_name_string<base>();
        return 4 + name.size;  // '/e_' + base_type_char + '_' + name  (minus the 1 char already in baseline)
    } else {
        return 0;
    }
}

// Detect at compile time whether an enum type has an operator| that returns the enum type itself
// (i.e. is a flag/bitmask enum). Unscoped enums implicitly convert to int so T|T works for all
// of them via integer promotion — we require the result type to be T to distinguish true flag enums.
template <typename T, typename = void>
struct is_flag_enum : std::false_type {};

template <typename T>
struct is_flag_enum<
    T,
    std::enable_if_t<std::is_enum_v<T> && std::is_same_v<T, decltype(std::declval<T>() | std::declval<T>())>>>
    : std::true_type {};

// Writes the type format info for a single type into the result string.
// Simple types: single character (e.g. 'I' for uint32_t)
// Regular enums: /e_<base_char>_<EnumTypeName>
// Flag enums (with operator|): /E_<base_char>_<EnumTypeName>
template <typename T, std::size_t MaxLen>
constexpr void write_type(helpers::static_string<MaxLen>& result) {
    using base = std::remove_cv_t<std::remove_reference_t<T>>;
    if constexpr (std::is_enum_v<base>) {
        constexpr auto enum_name = get_type_name_string<base>();
        result.push_back('/');
        result.push_back(is_flag_enum<base>::value ? 'E' : 'e');
        result.push_back('_');
        result.push_back(device_print_type<base>::value.type_char);
        result.push_back('_');
        for (std::size_t j = 0; j < enum_name.size; j++) {
            result.push_back(enum_name.data[j]);
        }
    } else {
        result.push_back(device_print_type<base>::value.type_char);
    }
}

// Finds the arg_index-th type in the Args... pack and calls write_type for it.
template <typename... Types>
struct TypeWriter;

template <>
struct TypeWriter<> {
    template <std::size_t MaxLen>
    static constexpr void write(helpers::static_string<MaxLen>&, uint32_t, uint32_t = 0) {}
};

template <typename First, typename... Rest>
struct TypeWriter<First, Rest...> {
    template <std::size_t MaxLen>
    static constexpr void write(helpers::static_string<MaxLen>& result, uint32_t arg_index, uint32_t current = 0) {
        if (current == arg_index) {
            write_type<First>(result);
        } else {
            TypeWriter<Rest...>::write(result, arg_index, current + 1);
        }
    }
};

// Main function to update format string with type information
// Supports both {} and {N} placeholder styles (fmtlib-compatible)
template <std::size_t N, typename... Args>
constexpr auto update_format_string(const char (&format)[N]) {
    constexpr std::size_t format_len = N - 1;  // Exclude null terminator

    // Calculate maximum result length.
    // Each {} placeholder (2 chars) expands to {N,T} where N is the arg index and T is the type char.
    // The net extra chars per placeholder = 2 + digits(max_index), where:
    //   - 1 digit  (N < 10):   net = 3
    //   - 2 digits (N < 100):  net = 4
    //   - 3 digits (N < 1000): net = 5
    // Use sizeof...(Args) to pick the right bound rather than always assuming 2.
    constexpr std::size_t num_args_ = sizeof...(Args);
    constexpr std::size_t max_index_digits_ = (num_args_ <= 9) ? 1 : (num_args_ <= 99) ? 2 : (num_args_ <= 999) ? 3 : 4;
    constexpr std::size_t enum_extra_ = (enum_extra_format_chars<Args>() + ... + 0);
    constexpr std::size_t result_len = format_len + (format_len / 2 + 1) * (2 + max_index_digits_) + enum_extra_;

    helpers::static_string<result_len> result;

    constexpr auto arg_reorder = get_arg_reorder<Args...>();

    std::size_t type_index = 0;

    for (std::size_t i = 0; i < format_len;) {
        parsing::FormatToken token = parsing::parse_format_token(format, i);

        if (token.type == parsing::TokenType::EscapedOpenBrace) {
            // Preserve escaped opening brace: {{
            result.push_back('{');
            result.push_back('{');
        } else if (token.type == parsing::TokenType::EscapedCloseBrace) {
            // Preserve escaped closing brace: }}
            result.push_back('}');
            result.push_back('}');
        } else if (token.type == parsing::TokenType::Placeholder) {
            // Determine the argument index for this placeholder
            uint32_t arg_index = token.is_indexed() ? token.index.value() : type_index++;

            // Unified handling for both indexed and non-indexed: output {index,type:format}
            result.push_back('{');

            // Output the index (updated with reordered index)
            auto reordered_index = arg_reorder[arg_index];
            for (size_t j = 0; j < arg_reorder.size(); j++) {
                if (arg_reorder[j] == arg_index) {
                    reordered_index = j;
                    break;
                }
            }
            result.push_back_uint32(reordered_index);

            // Add comma and type character (enum types emit extended type info)
            result.push_back(',');
            TypeWriter<Args...>::write(result, arg_index);

            // If there are any format specifiers, add them
            bool has_format_spec = token.fill.has_value() || token.align.has_value() || token.sign.has_value() ||
                                   token.use_alternate_form || token.use_zero_padding || token.width.has_value() ||
                                   token.precision.has_value() || token.use_current_locale ||
                                   token.format_type.has_value();
            if (has_format_spec) {
                result.push_back(':');
                if (token.fill.has_value()) {
                    result.push_back(token.fill.value());
                }
                if (token.align.has_value()) {
                    result.push_back(token.align.value());
                }
                if (token.sign.has_value()) {
                    result.push_back(token.sign.value());
                }
                if (token.use_alternate_form) {
                    result.push_back('#');
                }
                if (token.use_zero_padding) {
                    result.push_back('0');
                }
                if (token.width.has_value()) {
                    result.push_back_uint32(token.width.value());
                }
                if (token.precision.has_value()) {
                    result.push_back('.');
                    result.push_back_uint32(token.precision.value());
                }
                if (token.use_current_locale) {
                    result.push_back('L');
                }
                if (token.format_type.has_value()) {
                    result.push_back(token.format_type.value());
                }
            }
            result.push_back('}');
        } else {
            // Regular character
            result.push_back(format[i]);
        }

        i = token.end_pos;
    }

    return result;
}

// New function that returns constexpr updated format string from arguments
// This allows calling with actual arguments: update_format_string_from_args("format {}", arg1, arg2, ...)
template <std::size_t N, typename... Args>
constexpr auto update_format_string_from_args(const char (&format)[N], const Args&... args) {
    ((void)sizeof(args), ...);  // Suppress unused parameter warnings for all args
    return update_format_string<N, Args...>(format);
}

}  // namespace formatting

namespace locking {

#if !defined(ARCH_WORMHOLE)
volatile tt_l1_ptr std::atomic<uint32_t>& get_lock_atomic() {
#if !defined(ARCH_QUASAR) || defined(ENV_LLK_INFRA)
    return get_device_print_buffer()->aux.lock;
#else
    // Atomics require the cached L1 alias.
    return GET_MAILBOX_ADDRESS_DEV_CACHED(dprint_buf.buffer)->aux.lock;
#endif
}
#endif

// Called in every iteration of the waits. On an active ERISC the wait yields to the base firmware, a lot of code: with
// two wait loops (the lock spin and the wait for space, when the RISCs of the core share the lock) one out-of-line copy
// is smaller than two expansions. Empty on every other RISC.
#if defined(COMPILE_FOR_ERISC)
static __attribute__((noinline, unused)) void yield_to_base_firmware() { internal_::risc_context_switch(); }
#endif
DEVICE_PRINT_ALWAYS_INLINE void yield() {
#if defined(COMPILE_FOR_ERISC)
    if constexpr (DevicePrintBufferType::processor_count > 1) {
        yield_to_base_firmware();
    } else {
        internal_::risc_context_switch();
    }
#endif
}

// Takes the buffer lock and invalidates this RISC's L1 cache, so that the positions and data that other RISCs wrote
// are seen. The lock is needed only when more than one RISC prints into the buffer.
DEVICE_PRINT_ALWAYS_INLINE void acquire_lock() {
    if constexpr (DevicePrintBufferType::processor_count > 1) {
#if defined(ARCH_WORMHOLE)
        volatile uint32_t* lock_ptr = &(get_device_print_buffer()->aux.lock);

        while (true) {
        again:
            // Wait until lock is free (0)
            while (*lock_ptr != 0) {
                invalidate_l1_cache();
                yield();
#if DEVICE_PRINT_DISPATCH_ENABLED
                // dispatch_s is then the only drainer: a RISC that holds the lock may be waiting for space.
                ::device_print_dispatcher_execute_hook();
#endif
            }

            // Write risc_id to lock to attempt to acquire it
            *lock_ptr = PROCESSOR_INDEX + 1;  // Use 1-based index to avoid writing 0 which is the free state

            // Make sure the write has propagated and other riscs see the updated value: read it back processor_count
            // times, and once more to confirm that we have acquired the lock. A loop, not unrolled, to keep it small:
            // GCC's complete unrolling ignores `#pragma GCC unroll`, so the trip count is hidden from it instead.
            uint32_t reads = DevicePrintBufferType::processor_count + 1;
            asm("" : "+r"(reads));
            for (uint32_t repeat = 0; repeat < reads; ++repeat) {
                invalidate_l1_cache();
                if (*lock_ptr != PROCESSOR_INDEX + 1) {
                    goto again;
                }
            }
            break;  // Successfully acquired lock
        }
#else
        auto& lock_atomic = get_lock_atomic();

        while (lock_atomic.exchange(1) != 0) {
            // Failed to acquire lock, wait and try again
            invalidate_l1_cache();
            yield();
#if DEVICE_PRINT_DISPATCH_ENABLED
            // dispatch_s is then the only drainer: a RISC that holds the lock may be waiting for space.
            ::device_print_dispatcher_execute_hook();
#endif
        }
#endif
    }

    // After acquiring the lock, invalidate our L1 cache to ensure we see the most up-to-date data in the buffer
    invalidate_l1_cache();
}

// Nobody else takes the lock when a single RISC prints into the buffer, so there is nothing to release.
DEVICE_PRINT_ALWAYS_INLINE void release_lock() {
    if constexpr (DevicePrintBufferType::processor_count > 1) {
#if defined(ARCH_WORMHOLE)
        volatile uint32_t* lock_ptr = &(get_device_print_buffer()->aux.lock);

        asm volatile("" ::: "memory");
        *lock_ptr = 0;  // Release lock by setting to 0
        asm volatile("" ::: "memory");
#else
        auto& lock_atomic = get_lock_atomic();
        lock_atomic = 0;
#endif
    }
}

void update_kernel_finished() {
    volatile tt_l1_ptr DevicePrintBufferType* device_print_buffer = get_device_print_buffer();
#if PROCESSOR_INDEX_DEFINED
    constexpr auto processor_index = PROCESSOR_INDEX;
#else
    auto processor_index = internal_::get_hw_thread_idx();
#endif
    if (device_print_buffer->aux.risc_state[processor_index - DevicePrintBufferType::processor_offset] !=
        DevicePrintRiscCoreState::PrintingDisabled) {
        device_print_buffer->aux.risc_state[processor_index - DevicePrintBufferType::processor_offset] =
            DevicePrintRiscCoreState::KernelNotPrinted;
    }
}

void initialize_lock() {
#if defined(ARCH_WORMHOLE)
    volatile uint32_t* lock_ptr = &(get_device_print_buffer()->aux.lock);
    asm volatile("" ::: "memory");
    *lock_ptr = 0;  // Ensure lock starts in free state
    asm volatile("" ::: "memory");
#else
    auto& lock_atomic = get_lock_atomic();
    lock_atomic = 0;
#endif
}

// The host's magic values live in the write position (wpos) and the read position (rpos) of the buffer, where
// valid positions are multiples of 4 up to sizeof(data), so anything above sizeof(data) is a magic value, and the
// magic values are told apart by one bit instead of by comparing with each of them:
//   wpos == DEBUG_PRINT_SERVER_DISABLED_MAGIC: printing is off, don't wait and don't touch anything.
//   wpos == DEBUG_PRINT_SERVER_STARTING_MAGIC or rpos == DEVICE_PRINT_RESET_BUFFER_MAGIC: restart from position 0.
// While a writer waits for space, wpos carries DEVICE_PRINT_WRITE_STALL_FLAG, which is not that bit.
inline constexpr uint32_t disabled_magic_bit = 0x40000000;
static_assert((DEBUG_PRINT_SERVER_DISABLED_MAGIC & disabled_magic_bit) != 0, "disabled magic must have the bit");
static_assert((DEBUG_PRINT_SERVER_STARTING_MAGIC & disabled_magic_bit) == 0, "starting magic must not have the bit");
static_assert((DEVICE_PRINT_WRITE_STALL_FLAG & disabled_magic_bit) == 0, "stall flag must not be the bit");
static_assert(sizeof(DevicePrintBufferType::data) < disabled_magic_bit, "positions are below the bit");

// A constant that the compiler materializes where it is used: it doesn't keep it in a register across the wait loop,
// where the payload words of print_message() already take seven registers.
template <auto Value>
DEVICE_PRINT_ALWAYS_INLINE auto in_place() {
    decltype(Value) value;
    asm volatile("li %0, %1" : "=r"(value) : "i"(static_cast<int32_t>(Value)));
    return value;
}

// The wrap-around record: a header with the largest info id and nothing else. info_id is the top 16 bits of the
// header (is_kernel:1, risc_id:5, message_payload:10, info_id:16), which a constant expression can't read back from
// the union.
inline constexpr uint32_t wrap_header_value = structures::DevicePrintHeader::max_info_id_value << 16;

// Waits until a message of `size` bytes can be written without catching up with the reader and returns the offset in
// the data where it goes (0 when printing is disabled). A RISC's first print after a kernel starts is preceded by a
// header-only kernel-id record that tells the host which kernel's ELF decodes this RISC's messages: it is written and
// published here, as a message of its own, before the wait for the space of the message.
DEVICE_PRINT_ALWAYS_INLINE uint32_t wait_for_space(
    volatile tt_l1_ptr DevicePrintBufferType* device_print_buffer,
    uint32_t size,
    [[maybe_unused]] uint32_t processor_index) {
    uint32_t data_size = sizeof(device_print_buffer->data);
    DEVICE_PRINT_REGISTER_HINT("+r"(data_size));  // one register for all the comparisons

    // The space to wait for: the kernel-id record first, when there is one, then the message.
    uint32_t reserve = size;
#if defined(KERNEL_BUILD) && !defined(ENV_LLK_INFRA)
    uint32_t announce = 0;
    volatile tt_l1_ptr DevicePrintRiscCoreState* risc_state =
        &device_print_buffer->aux.risc_state[processor_index - DevicePrintBufferType::processor_offset];
    if (*risc_state == DevicePrintRiscCoreState::KernelNotPrinted) {
        uint32_t launch_idx = *GET_MAILBOX_ADDRESS_DEV(launch_msg_rd_ptr);
        tt_l1_ptr launch_msg_t* const launch_msg = GET_MAILBOX_ADDRESS_DEV(launch[launch_idx]);
        auto kernel_id = launch_msg->kernel_config.watcher_kernel_ids[processor_index];
        structures::DevicePrintHeader new_kernel_message = {};
        new_kernel_message.is_kernel = 1;
        new_kernel_message.risc_id = processor_index;
        new_kernel_message.message_payload = structures::DevicePrintHeader::max_message_payload_size;
        new_kernel_message.info_id = kernel_id;
        announce = new_kernel_message.value;
        reserve = sizeof(uint32_t);
    }
#endif

    uint32_t write_position;
    while (true) {
        write_position = device_print_buffer->aux.wpos;
        uint32_t read_position = device_print_buffer->aux.rpos;
        if (write_position > data_size) {
            if ((write_position & disabled_magic_bit) != 0) {
                // If we are in disabled state, return immediately without waiting for space.
                return 0;
            }
            // Print server starting magic: initialize valid state, the same as for a reset.
            read_position = write_position;
        }

        // The record does not fit until the end of the buffer: it goes at the start, after a wrap-around record.
        bool wrap = write_position + reserve > data_size;
        bool stalled = false;
        while (true) {
            if (read_position > data_size) {
                // Reader is suggesting buffer clear (usually at startup), reset buffer state to empty to avoid waiting
                // for the reader to move.
                device_print_buffer->aux.wpos = 0;
                device_print_buffer->aux.rpos = 0;
                write_position = 0;
                stalled = false;  // the store of 0 above cleared the stall flag: no second store of wpos
                break;
            }
            if (wrap) {
                // It is important not to perform wrap around while reader position is at the beginning of the buffer,
                // as it will be the same state as at the beginning when buffer is empty. So in order to distinguish
                // real empty state and wrap around state, we wait for reader to progress from start. The reader must
                // also be done with the data until the end of the buffer (rpos <= wpos) before we wrap around.
                if (read_position - 1 < write_position) {
                    if (write_position < data_size) {
                        // We can fit a wrap around message, write it now so reader can process it while we wait for
                        // space.
                        auto value = in_place<wrap_header_value>();
                        *reinterpret_cast<device_print_buffer_ptr<decltype(value)>>(
                            device_print_buffer->data + write_position) = value;
                    }
                    // Wrap around to the beginning of the buffer and continue waiting for space there. The store of 0
                    // clears the stall flag: the wait for the reader is over (a new stall there is a new wait).
                    if (stalled) {
                        WAYPOINT("DPD");
                    }
                    write_position = device_print_buffer->aux.wpos = 0;
                    wrap = false;
                    stalled = false;
                    continue;
                }
            } else if (read_position - write_position - 1 >= reserve) {
                // Not (write_position < read_position <= write_position + reserve): enough space
                break;
            }

            if (!stalled) {
                // Mark that we are in stall waiting for reader
                device_print_buffer->aux.wpos = write_position | in_place<DEVICE_PRINT_WRITE_STALL_FLAG>();
                WAYPOINT("DPW");
                stalled = true;
            }
            invalidate_l1_cache();
            yield();
#if DEVICE_PRINT_DISPATCH_ENABLED
            ::device_print_dispatcher_execute_hook();
#endif
            // If we've closed the device, we've now disabled printing on it, don't hang.
            if ((device_print_buffer->aux.wpos & disabled_magic_bit) != 0) {
                return 0;
            }
            // Read new read position for next check
            read_position = device_print_buffer->aux.rpos;
        }

        if (stalled) {
            WAYPOINT("DPD");
            // Clear stall state, we can continue with normal flow now.
            device_print_buffer->aux.wpos = write_position;
        }
#if defined(KERNEL_BUILD) && !defined(ENV_LLK_INFRA)
        if (announce == 0) {
            break;
        }
        // Write the kernel-id record and move the write position past it, then wait for the space of the message.
        *reinterpret_cast<device_print_buffer_ptr<uint32_t>>(device_print_buffer->data + write_position) = announce;
        device_print_buffer->aux.wpos = write_position + sizeof(uint32_t);
        *risc_state = in_place<DevicePrintRiscCoreState::KernelPrinted>();
        announce = 0;
        reserve = size;
#else
        break;
#endif
    }
    return write_position;
}

}  // namespace locking

namespace serialization {

// Offsets of the arguments in the payload (the image has no header).
template <typename... Args>
constexpr std::array<uint32_t, sizeof...(Args)> get_payload_offsets() {
    std::array<uint32_t, sizeof...(Args)> offsets = formatting::get_arg_offsets<Args...>();
    for (auto& offset : offsets) {
        offset -= sizeof(structures::DevicePrintHeader::value);
    }
    return offsets;
}

// Size of a message (header + arguments, padded to 4 bytes) for the argument types Args.
template <typename... Args>
constexpr uint32_t get_total_message_size() {
    constexpr auto type_infos = formatting::get_types_info<Args...>();
    uint32_t total_size = sizeof(structures::DevicePrintHeader::value);  // Start with header size
    for (size_t i = 0; i < sizeof...(Args); ++i) {
        total_size += type_infos[i].size_in_bytes;
    }
    if (total_size % 4 != 0) {
        total_size += 4 - (total_size % 4);  // Pad to next multiple of 4 bytes
    }
    return total_size;
}

// The scalars: the arguments that a print can pass as payload words (see print_words).
template <typename T>
inline constexpr bool word_argument =
    std::is_integral_v<T> || std::is_enum_v<T> || std::is_pointer_v<T> || std::is_same_v<T, float> ||
    std::is_same_v<T, double> || std::is_same_v<T, ct_string> || std::is_same_v<T, bf4_t> || std::is_same_v<T, bf8_t> ||
    std::is_same_v<T, bf16_t> || std::is_same_v<T, dp_top_callstack_t>;
template <typename T>
inline constexpr bool word_argument<dp_type_name_t<T>> = true;
template <>
inline constexpr bool word_argument<dp_typed_array_t<0>> = true;

template <typename... Args>
inline constexpr bool all_word_arguments = (word_argument<Args> && ...);

// The bytes of a scalar argument, as an unsigned integer of its size (narrow values zero-extended to 32 bits).
template <typename T>
DEVICE_PRINT_ALWAYS_INLINE auto argument_bits(const T& argument) {
    if constexpr (std::is_same_v<T, bool>) {
        return static_cast<uint32_t>(argument);
    } else if constexpr (std::is_enum_v<T>) {
        return argument_bits(static_cast<std::underlying_type_t<T>>(argument));
    } else if constexpr (std::is_integral_v<T>) {
        if constexpr (sizeof(T) <= sizeof(uint32_t)) {
            return static_cast<uint32_t>(static_cast<std::make_unsigned_t<T>>(argument));
        } else {
            return static_cast<uint64_t>(argument);
        }
    } else if constexpr (std::is_same_v<T, float>) {
        return __builtin_bit_cast(uint32_t, argument);
    } else if constexpr (std::is_same_v<T, double>) {
        return __builtin_bit_cast(uint64_t, argument);
    } else if constexpr (std::is_pointer_v<T>) {
        return reinterpret_cast<std::uintptr_t>(argument);
    } else if constexpr (std::is_same_v<T, ct_string>) {
        return reinterpret_cast<std::uintptr_t>(argument.ptr);
    } else {
        static_assert(std::is_same_v<T, bf4_t> || std::is_same_v<T, bf8_t> || std::is_same_v<T, bf16_t>);
        return static_cast<uint32_t>(static_cast<uint16_t>(argument.val));
    }
}
template <typename T>
DEVICE_PRINT_ALWAYS_INLINE std::uintptr_t argument_bits(const dp_type_name_t<T>&) {
    return reinterpret_cast<std::uintptr_t>(register_type_name_string<T>());
}
DEVICE_PRINT_ALWAYS_INLINE uint32_t argument_bits(const dp_typed_array_t<0>& argument) { return argument.type; }

// Word K of an argument's bytes.
template <uint32_t K, typename T>
DEVICE_PRINT_ALWAYS_INLINE uint32_t argument_word(const T& argument) {
    if constexpr (std::is_same_v<T, dp_top_callstack_t>) {
        constexpr uint32_t words_per_pointer = sizeof(std::uintptr_t) / sizeof(uint32_t);
        constexpr uint32_t index = K / words_per_pointer;
        const std::uintptr_t value = index == 0 ? argument.pc : (index == 1 ? argument.ra : argument.skip_frames);
        return static_cast<uint32_t>(static_cast<uint64_t>(value) >> (32 * (K % words_per_pointer)));
    } else {
        return static_cast<uint32_t>(static_cast<uint64_t>(argument_bits(argument)) >> (32 * K));
    }
}

// What an argument at payload offset OFFSET of SIZE bytes contributes to payload word W. Arguments of 4 bytes and more
// are word-aligned (largest first, sizes multiples of 4), and a print sent as words has no two narrow arguments in one
// word.
template <uint32_t W, uint32_t OFFSET, uint32_t SIZE, typename T>
DEVICE_PRINT_ALWAYS_INLINE uint32_t word_part(const T& argument) {
    if constexpr (W >= OFFSET / 4 && W < (OFFSET + SIZE + 3) / 4) {
        return argument_word<W - OFFSET / 4>(argument);
    } else {
        return 0;
    }
}

template <typename... Args>
struct payload_layout {
    static constexpr std::array<uint32_t, sizeof...(Args)> offsets = get_payload_offsets<Args...>();
    static constexpr std::array<formatting::device_print_type_info, sizeof...(Args)> types =
        formatting::get_types_info<Args...>();
};

// Payload word W of a print.
template <uint32_t W, typename... Args, std::size_t... Is>
DEVICE_PRINT_ALWAYS_INLINE uint32_t payload_word(std::index_sequence<Is...>, const Args&... args) {
    using layout = payload_layout<Args...>;
    return (word_part<W, layout::offsets[Is], layout::types[Is].size_in_bytes>(args) | ... | 0u);
}

// The narrow arguments (1 and 2 bytes) that share a payload word with the one before them. The payload holds the
// arguments largest first, so the narrow ones are packed at its end and none of them straddles a word.
template <typename... Args>
constexpr uint32_t merged_arguments() {
    constexpr uint32_t sizes[] = {formatting::get_type_info<Args>().size_in_bytes..., 4};
    uint32_t narrow = 0, narrow_bytes = 0;
    for (uint32_t i = 0; i < sizeof...(Args); ++i) {
        if (sizes[i] < sizeof(uint32_t)) {
            ++narrow;
            narrow_bytes += sizes[i];
        }
    }
    return narrow - (narrow_bytes + 3) / 4;
}

// Stores the arguments, all but argument Skip, at their payload offsets (template arguments: constants also at -O0):
// the scalars and TileSlices (Scalars), then the typed arrays (Arrays), so that no value is kept across a copy call.
template <typename To, bool Arrays, uint32_t Offset, bool Write, typename T>
DEVICE_PRINT_ALWAYS_INLINE void fill_argument(
    [[maybe_unused]] typename To::byte_ptr dst, [[maybe_unused]] const T& argument) {
    if constexpr (Write && formatting::is_typed_array<T> == Arrays) {
        formatting::device_print_type<T>::template serialize<To>(dst, Offset, argument);
    }
}

template <typename To, std::size_t Skip, bool Scalars, bool Arrays, typename... Args, std::size_t... Is>
DEVICE_PRINT_ALWAYS_INLINE void fill_payload_impl(
    typename To::byte_ptr dst, std::index_sequence<Is...>, const Args&... args) {
    using layout = payload_layout<Args...>;
    (fill_argument < To, false, layout::offsets[Is], Scalars && Is != Skip > (dst, args), ...);
    (fill_argument < To, true, layout::offsets[Is], Arrays && Is != Skip > (dst, args), ...);
}

template <typename To, std::size_t Skip = ~std::size_t{0}, bool Scalars = true, bool Arrays = true, typename... Args>
DEVICE_PRINT_ALWAYS_INLINE void fill_payload(typename To::byte_ptr dst, const Args&... args) {
    fill_payload_impl<To, Skip, Scalars, Arrays, Args...>(dst, std::index_sequence_for<Args...>{}, args...);
}

// Whether every 8-byte scalar is 8-byte aligned in the payload (see formatting::to_store_image).
template <typename... Args>
constexpr bool wide_values_aligned() {
    constexpr std::array<uint32_t, sizeof...(Args)> offsets = get_payload_offsets<Args...>();
    constexpr bool wide[] = {(word_argument<Args> && formatting::get_type_info<Args>().size_in_bytes % 8 == 0)...};
    for (std::size_t i = 0; i < sizeof...(Args); ++i) {
        if (wide[i] && offsets[i] % 8 != 0) {
            return false;
        }
    }
    return true;
}
}  // namespace serialization

// The library: print_message() takes the lock, waits for space, writes the header and the payload, and returns the
// address after the message; print_end() publishes it and releases the lock. The low 4 bits of the call site's 16-byte
// aligned DevicePrintStringInfo entry carry the payload size n: n = 0..14 words follow as XLEN-sized arguments (a1-a7,
// then the caller's outgoing argument area), or count_form: print_message(entry + count_form, n, image).
// print_message() is a variadic alias of device_print_message(), whose named parameters take all eight argument
// registers.
inline constexpr uint32_t count_form = 15;

extern "C" {
static DEVICE_PRINT_LIBRARY_FUNCTION DEVICE_PRINT_NO_LOOP_SPLIT std::uintptr_t device_print_message(
    std::uintptr_t info_address_and_words,
    std::uintptr_t w0,
    std::uintptr_t w1,
    std::uintptr_t w2,
    std::uintptr_t w3,
    std::uintptr_t w4,
    std::uintptr_t w5,
    std::uintptr_t w6,
    ...) {
    static_assert(sizeof(std::uintptr_t) == 4 || sizeof(std::uintptr_t) == 8, "XLEN must be 32 or 64");
    const uint32_t words_in_entry = info_address_and_words & 15;
    const uint32_t payload_words = words_in_entry != count_form ? words_in_entry : static_cast<uint32_t>(w0);

    volatile tt_l1_ptr DevicePrintBufferType* device_print_buffer = get_device_print_buffer();
#if PROCESSOR_INDEX_DEFINED
    constexpr auto processor_index = PROCESSOR_INDEX;
#else
    const auto processor_index = internal_::get_hw_thread_idx();
#endif
    locking::acquire_lock();
    const uint32_t write_position = locking::wait_for_space(
        device_print_buffer, sizeof(uint32_t) + payload_words * sizeof(uint32_t), processor_index);

    // The entry's index is its distance from the start of the section; the low bits that carry the payload size fall
    // off the division. It takes 65536 call sites in one ELF (1 MB of entries) to exceed max_info_id_value.
    using DevicePrintHeaderType = structures::DevicePrintHeader;
    std::uintptr_t string_info_start_address = reinterpret_cast<std::uintptr_t>(__device_print_strings_info_start);
    DevicePrintHeaderType header = {};
    header.is_kernel = DEVICE_PRINT_IS_KERNEL;
    header.risc_id = processor_index;
    header.message_payload = payload_words * sizeof(uint32_t);
    header.info_id = static_cast<uint32_t>(
        (info_address_and_words - string_info_start_address) / sizeof(structures::DevicePrintStringInfo));

    // Write the header and copy the payload after it: the words in registers first (one register at a time, the
    // others move down), then the others from memory (the caller's stack or the image).
    auto message =
        reinterpret_cast<device_print_buffer_ptr<uint32_t>>(&(device_print_buffer->data[0]) + write_position);
    message[0] = header.value;
    auto payload = message + 1;
    const auto payload_end = payload + payload_words;
#if defined(__clang__)
    // clang (clangd, kernel clang-tidy) has no __builtin_next_arg; va_start gives the same address.
    __builtin_va_list stack_words;
    __builtin_va_start(stack_words, w6);
    const uint8_t* memory = reinterpret_cast<const uint8_t*>(stack_words);
    __builtin_va_end(stack_words);
#else
    const uint8_t* memory = reinterpret_cast<const uint8_t*>(__builtin_next_arg(w6));
#endif
    uint32_t stride = sizeof(std::uintptr_t);
    uint32_t in_registers = 7;
    if (words_in_entry == count_form) {
        memory = reinterpret_cast<const uint8_t*>(w1);
        stride = sizeof(uint32_t);
        in_registers = 0;
    }
    const auto registers_end = payload + (payload_words < in_registers ? payload_words : in_registers);
#pragma GCC unroll 1
    while (payload != registers_end) {
        *payload++ = w0;
        w0 = w1;
        w1 = w2;
        w2 = w3;
        w3 = w4;
        w4 = w5;
        w5 = w6;
    }
#pragma GCC unroll 1
    while (payload != payload_end) {
        *payload++ = *reinterpret_cast<const formatting::image_word_t*>(memory);
        memory += stride;
    }
    return reinterpret_cast<std::uintptr_t>(payload_end);
}
}

// The declaration that call sites use (see above). GCC warns about an alias of a different type.
#if !defined(__clang__)
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wattribute-alias"
#endif
static std::uintptr_t print_message(std::uintptr_t info_address_and_words, ...)
    __attribute__((alias("device_print_message"), unused));
#if !defined(__clang__)
#pragma GCC diagnostic pop
#endif

// Publishes the message (moves the write position to `end`, print_message()'s return value) unless the host disabled
// printing meanwhile (a write position past the data), and releases the lock.
static DEVICE_PRINT_LIBRARY_FUNCTION void print_end(std::uintptr_t end) {
    volatile tt_l1_ptr DevicePrintBufferType* device_print_buffer = get_device_print_buffer();
    if (device_print_buffer->aux.wpos <= sizeof(device_print_buffer->data)) {
        device_print_buffer->aux.wpos = end - reinterpret_cast<std::uintptr_t>(&(device_print_buffer->data[0]));
    }
    locking::release_lock();
}

// The entries stay 16-byte aligned (aligned(16), sizeof() apart, the section starts at 0x6500000 with
// __device_print_strings_info_start); the info id is an entry's distance from that symbol divided by the entry size.
static_assert(
    sizeof(structures::DevicePrintStringInfo) % 16 == 0, "DevicePrintStringInfo entries must keep 16-byte alignment");

// The store form (see send_message): n - 1 for an image of n words (1..256) in the free low bits of the entry (high 4
// bits) and of the 16-byte-aligned image (low 4 bits), so that a call site sets two argument registers.
static DEVICE_PRINT_LIBRARY_FUNCTION std::uintptr_t device_print_store(
    std::uintptr_t info_address_and_words, std::uintptr_t image_and_words) {
    const uint32_t words = ((info_address_and_words & 15) << 4 | (image_and_words & 15)) + 1;
    const std::uintptr_t image = image_and_words & ~static_cast<std::uintptr_t>(15);
    return print_message(info_address_and_words | count_form, words, image);
}

// The typed-array form (see send_message): the entry plus the payload words before the first typed array (0..15), its
// first word (len << 16 | type, len 1..255) with the payload's words in the top 8 bits, and its elements, which are
// copied under the lock (after the L1 invalidation); the call site writes the other arguments before print_end().
static DEVICE_PRINT_LIBRARY_FUNCTION std::uintptr_t device_print_typed_array(
    std::uintptr_t info_address_and_offset, std::uintptr_t first_word_and_words, std::uintptr_t elements) {
    const uint32_t first_word = static_cast<uint32_t>(first_word_and_words) & 0xffffff;
    const uint32_t len = first_word >> 16;
    const uint32_t words = static_cast<uint32_t>(first_word_and_words) >> 24;
    const std::uintptr_t data = reinterpret_cast<std::uintptr_t>(&(get_device_print_buffer()->data[0]));
    const std::uintptr_t end = print_message(info_address_and_offset | count_form, words, data);
    auto destination = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(end) - words +
                       static_cast<uint32_t>(info_address_and_offset & 15);
    destination[0] = first_word;
    const uint32_t* source = reinterpret_cast<const uint32_t*>(elements);
#pragma GCC unroll 1
    for (uint32_t i = 0; i < len; ++i) {
        destination[i + 1] = source[i];
    }
    return end;
}

// The words form: print_message(entry + n, word 0, ..., word n-1).
template <typename... Args, std::size_t... Ws>
DEVICE_PRINT_ALWAYS_INLINE std::uintptr_t print_words(
    std::uintptr_t info_address, std::index_sequence<Ws...>, const Args&... args) {
    return print_message(
        info_address + sizeof...(Ws), serialization::payload_word<Ws>(std::index_sequence_for<Args...>{}, args...)...);
}

// The widest print of scalar arguments, in payload words (see print_argument_bytes); the store form holds them all.
inline constexpr uint32_t store_form_max_words = 64;

// The typed array that comes first in the payload.
template <typename... Args>
constexpr std::size_t first_typed_array() {
    constexpr std::array<uint32_t, sizeof...(Args)> offsets = serialization::get_payload_offsets<Args...>();
    constexpr bool arrays[] = {formatting::is_typed_array<Args>...};
    std::size_t first = 0;
    for (std::size_t i = 0; i < sizeof...(Args); ++i) {
        if (arrays[i] && (!arrays[first] || offsets[i] < offsets[first])) {
            first = i;
        }
    }
    return first;
}

// How a print hands its payload to the library (see send_message).
enum class print_form { none, words, store, typed_array, lone_tile_slice, reserve };

// The typed-array form keeps a print's other values in registers across the library call. A print with more than 8
// words of them (on RV64, more than 2 words next to an 8-byte value, which an image takes with one store) stores them
// into an image first: the store form, whose typed arrays are copied after the call.
template <typename... Args>
constexpr uint32_t typed_array_form_max_other_words() {
    constexpr bool wide =
        ((serialization::word_argument<Args> && formatting::get_type_info<Args>().size_in_bytes % 8 == 0) || ...);
    return sizeof(std::uintptr_t) == 8 && wide ? 2 : 8;
}

template <typename... Args>
constexpr print_form form_of() {
    constexpr uint32_t words =
        (serialization::get_total_message_size<Args...>() - sizeof(structures::DevicePrintHeader::value)) / 4;
    constexpr bool tile_slices = (formatting::is_tile_slice<Args> || ...);
    if (words == 0) {
        return print_form::none;
    }
    if constexpr ((formatting::is_typed_array<Args> || ...)) {
        constexpr std::size_t first = first_typed_array<Args...>();
        constexpr uint32_t lengths[] = {formatting::typed_array_length<Args>...};
        constexpr uint32_t array_words =
            ((formatting::is_typed_array<Args> ? formatting::typed_array_length<Args> + 1 : 0) + ...);
        if (lengths[first] <= 255 && serialization::get_payload_offsets<Args...>()[first] / 4 <= 15 &&
            words - array_words <= typed_array_form_max_other_words<Args...>()) {
            return print_form::typed_array;
        }
        return words <= store_form_max_words ? print_form::store : print_form::reserve;
    }
    if (sizeof...(Args) == 1 && tile_slices) {
        return print_form::lone_tile_slice;
    }
#if defined(__OPTIMIZE__)
    // At -O0 nothing folds the computation of the words, which costs far more code than an image. On RV64 a print with
    // 8-byte values whose words don't all fit the argument registers is stored (one store per value) rather than
    // passed as words (two per value, on the stack).
    constexpr bool wide = ((formatting::get_type_info<Args>().size_in_bytes == 8) || ...);
    if (serialization::all_word_arguments<Args...> && words < count_form &&
        serialization::merged_arguments<Args...>() == 0 && !(sizeof(std::uintptr_t) == 8 && words > 7 && wide)) {
        return print_form::words;
    }
#endif
    return words <= store_form_max_words ? print_form::store : print_form::reserve;
}

// The payload bytes of a print's scalar arguments (all but typed arrays and TileSlices; small ones share words).
template <typename... Args>
constexpr uint32_t scalar_argument_bytes() {
    constexpr uint32_t bytes =
        ((formatting::get_type_info<Args>().type_char == 'A' || formatting::is_tile_slice<Args>
              ? 0u
              : formatting::get_type_info<Args>().size_in_bytes) +
         ... + 0u);
    return (bytes + 3) / 4 * 4;
}

// The scalar arguments of one print take at most 256 bytes, so that the store form holds every scalar print; the
// compiler names the size in print_argument_bytes<Bytes> and in the failed comparison.
template <uint32_t Bytes>
struct print_argument_bytes {
    static_assert(
        Bytes <= 4 * store_form_max_words,
        "DEVICE_PRINT: the arguments of this print take more than 256 bytes (Bytes in print_argument_bytes<Bytes>); "
        "split it into several DEVICE_PRINT calls, or print an array as a typed array");
    static constexpr bool value = true;
};

// At most 64 arguments (a typed array or a TileSlice is one); the compiler names the count in print_argument_count.
inline constexpr uint32_t max_print_arguments = 64;
template <uint32_t Count>
struct print_argument_count {
    static_assert(
        Count <= max_print_arguments,
        "DEVICE_PRINT: this print has more than 64 arguments (Count in print_argument_count<Count>); split it into "
        "several DEVICE_PRINT calls");
    static constexpr bool value = true;
};

// Argument I of a print.
template <std::size_t I, typename T, typename... Rest>
DEVICE_PRINT_ALWAYS_INLINE const auto& nth_argument(const T& first, const Rest&... rest) {
    if constexpr (I == 0) {
        return first;
    } else {
        return nth_argument<I - 1>(rest...);
    }
}

// Sends one message in one of these forms, chosen by the argument types (no argument expression runs under the lock):
//  - none: print_message(entry). words: print_message(entry + n, word 0, ..., word n-1), up to 14 unshared words.
//  - store: one store per argument into a 16-byte-aligned image on the stack, passed to device_print_store(); the other
//    prints of up to 64 words; their typed arrays are copied after the call.
//  - typed_array: device_print_typed_array() copies the first typed array (at most 15 words after the start, at most 8
//    words of other values) under the lock; the call site writes the other arguments.
//  - lone_tile_slice: the print's copy of the TileSlice is the image. reserve: the other prints (more than 64 words):
//    the print buffer's data as a dummy image, then the call site writes the payload.
// Then print_end().
template <typename Tag, typename... Args>
DEVICE_PRINT_ALWAYS_INLINE void send_message(Args&... args) {
    constexpr uint32_t payload =
        serialization::get_total_message_size<Args...>() - sizeof(structures::DevicePrintHeader::value);
    {
        // The header's payload field is 10 bits: a bigger print doesn't compile (-Werror=overflow).
        structures::DevicePrintHeader range_check = {};
        range_check.message_payload = payload;
        (void)range_check;
    }
    // wait_for_space() can make room for a message only if the message fits in the data area of the print buffer; a
    // bigger one would be written past its end and then stall this RISC for good.
    static_assert(
        serialization::get_total_message_size<Args...>() <= sizeof(DevicePrintBufferType::data),
        "DEVICE_PRINT: the message does not fit in the print buffer of this RISC; print fewer or smaller arguments");
    static_assert(print_argument_count<sizeof...(Args)>::value);
    static_assert(print_argument_bytes<scalar_argument_bytes<Args...>()>::value);
    const std::uintptr_t info_address = register_string_info<Tag, Args...>();

    constexpr uint32_t words = payload / sizeof(uint32_t);
    constexpr print_form form = form_of<Args...>();
    std::uintptr_t end;  // the address after the message (see print_message)
    if constexpr (form == print_form::none) {
        // Every argument has a size, except a type without a serialization (size 0): name its serializer, whose
        // static_assert reports it ("No serialization defined for this type").
        ((void)&formatting::device_print_type<Args>::template serialize<formatting::to_image>, ...);
        end = print_message(info_address);
    } else if constexpr (form == print_form::words) {
        end = print_words(info_address, std::make_index_sequence<words>{}, args...);
    } else if constexpr (form == print_form::store) {
        // The scalars and TileSlices into the image; typed arrays (their slots in the image are not written) after the
        // call, into the message.
        alignas(16) formatting::image_word_t image[words];
        using to = std::conditional_t<
            !(formatting::is_tile_slice<Args> || ...) && serialization::wide_values_aligned<Args...>(),
            formatting::to_store_image,
            formatting::to_image>;
        serialization::fill_payload<to, ~std::size_t{0}, true, false>(
            reinterpret_cast<volatile uint8_t*>(image), args...);
        const std::uintptr_t image_and_words = reinterpret_cast<std::uintptr_t>(image) + ((words - 1) & 15);
        end = device_print_store(info_address + ((words - 1) >> 4), image_and_words);
        if constexpr ((formatting::is_typed_array<Args> || ...)) {
            serialization::fill_payload<formatting::to_buffer, ~std::size_t{0}, false, true>(
                reinterpret_cast<formatting::to_buffer::byte_ptr>(end - payload), args...);
        }
    } else if constexpr (form == print_form::typed_array) {
        constexpr std::size_t first = first_typed_array<Args...>();
        constexpr uint32_t before = serialization::get_payload_offsets<Args...>()[first] / sizeof(uint32_t);
        const auto& array = nth_argument<first>(args...);
        using array_type = std::remove_cv_t<std::remove_reference_t<decltype(array)>>;
        end = device_print_typed_array(
            info_address + before,
            formatting::device_print_type<array_type>::first_word(array) | (words << 24),
            reinterpret_cast<std::uintptr_t>(array.ptr));
        if constexpr (sizeof...(Args) > 1) {
            serialization::fill_payload<formatting::to_buffer, first>(
                reinterpret_cast<formatting::to_buffer::byte_ptr>(end - payload), args...);
        }
    } else if constexpr (form == print_form::lone_tile_slice) {
        end = print_message(
            info_address + count_form, words, reinterpret_cast<const uint32_t*>(formatting::tile_slice_image(args...)));
    } else {
        const std::uintptr_t data = reinterpret_cast<std::uintptr_t>(&(get_device_print_buffer()->data[0]));
        end = print_message(info_address + count_form, words, data);
        serialization::fill_payload<formatting::to_buffer>(
            reinterpret_cast<formatting::to_buffer::byte_ptr>(end - payload), args...);
    }
    print_end(end);
}

}  // namespace device_print_detail

#undef DEVICE_PRINT_ALWAYS_INLINE
#undef DEVICE_PRINT_LIBRARY_FUNCTION
#undef DEVICE_PRINT_NO_LOOP_SPLIT
#undef DEVICE_PRINT_REGISTER_HINT

#else

#define DEVICE_PRINT(format, ...) \
    do {                          \
    } while (0)
#define DEVICE_PRINT_INITIALIZE_LOCK()
#define DEVICE_PRINT_KERNEL_FINISHED()
#define DEVICE_PRINT_FORCE_NOINLINE

#endif
