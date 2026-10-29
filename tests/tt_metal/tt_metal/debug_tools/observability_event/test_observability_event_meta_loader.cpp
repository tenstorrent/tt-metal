// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0
//
// Observability event metadata, layer 1b: the host loader's record walk (tt::llrt::parse_meta_records) on
// .tt_observability_event_meta / .tt_observability_event_str bytes built here, with no ELF, no device and no device
// compiler. Valid sections are written field by field the way emit_event() (hostdev/observability_event_meta.h) writes
// them; damaged and stale ones are cut or hand-built from those. The walk must return exactly the records it can read,
// never read outside the two sections, and stop at the first record it cannot size. The damage tests place every input
// against an unreadable guard page, so a read past the end of either section crashes the test rather than passing
// unseen.

#include <gtest/gtest.h>

#include <cstddef>
#include <cstdint>
#include <cstring>
#include <limits>
#include <random>
#include <span>
#include <stdexcept>
#include <string_view>
#include <vector>

#include <sys/mman.h>
#include <unistd.h>

#include "hostdev/observability_event_meta.h"
#include "llrt/observability_event_meta.hpp"

namespace observability_event_loader_test {

using namespace tt::observability_event;
using namespace tt::observability_event::detail;
using tt::llrt::parse_meta_records;
using tt::llrt::ParsedMetaRecords;

enum class Level : int8_t { Low = -1, High = 7 };
enum Wide : uint64_t { kWideMax = std::numeric_limits<uint64_t>::max() };

struct Style {
    uint32_t color;
    uint8_t verbosity;
};
struct Nested {
    const char* name;
    Style style;
    uint16_t flags;
};
struct AllWidths {
    const char* name;
    bool on;
    int8_t i8;
    uint16_t u16;
    int32_t i32;
    int64_t i64;
    Level level;
    Wide wide;
};
struct Defaulted {
    const char* name;
    uint32_t color = 0x123456;
    uint8_t verbosity = 4;
};
struct TwoStrings {
    const char* name;
    const char* detail;
};

// The two sections as the device's emitter lays them out. .tt_observability_event_str sits at kStrVma, as it would in
// the image.
class Sections {
public:
    static constexpr uint64_t kStrVma = 0x06600000;

    uint32_t str(std::string_view s) {
        const auto ptr = static_cast<uint32_t>(kStrVma + str_.size());
        for (char c : s) {
            str_.push_back(static_cast<std::byte>(c));
        }
        str_.push_back(std::byte{0});
        return ptr;
    }
    void align(size_t a) {
        while (meta_.size() % a != 0) {
            meta_.push_back(std::byte{0});
        }
    }
    // A field of BYTES little-endian bytes, aligned to min(BYTES, 4) like the emitter's .balign.
    void put(uint64_t bits, size_t bytes) {
        align(bytes < 4 ? bytes : 4);
        for (size_t i = 0; i < bytes; i++) {
            meta_.push_back(static_cast<std::byte>(bits >> (8 * i)));
        }
    }
    void header(uint32_t id, std::string_view signature, std::string_view file = "kernel.cpp", uint32_t line = 1) {
        align(4);
        put(id, 4);
        put(str(signature), 4);
        put(str(file), 4);
        put(line, 4);
    }
    // VALUE's whole record: the header, then each leaf field, then padding to 4.
    template <class T>
    void record(uint32_t id, const T& value, std::string_view file = "kernel.cpp", uint32_t line = 1) {
        header(id, signature_of<T>(), file, line);
        for_each_leaf(value, [&](const auto& field) {
            using F = field_t<decltype(field)>;
            constexpr char code = field_code<F>();
            if constexpr (code == 's') {
                put(str(field), 4);
            } else {
                put(field_bits(field), code_bytes(code));
            }
        });
        align(4);
    }
    // The pre-metadata 16-byte record {id, name, file, line} that a stale JIT cache can still hold.
    void old_record(uint32_t id, std::string_view name) { header(id, name); }

    std::vector<std::byte>& meta() { return meta_; }
    std::vector<std::byte>& strs() { return str_; }
    ParsedMetaRecords parse() const { return parse_meta_records(meta_, str_, kStrVma); }

private:
    std::vector<std::byte> meta_;
    std::vector<std::byte> str_;
};

// A copy of BYTES that ends exactly where an unreadable page begins: reading one byte past it faults.
class Guarded {
public:
    explicit Guarded(std::span<const std::byte> bytes) {
        const size_t page = static_cast<size_t>(::sysconf(_SC_PAGESIZE));
        map_bytes_ = (bytes.size() + page - 1) / page * page + page;
        void* p = ::mmap(nullptr, map_bytes_, PROT_READ | PROT_WRITE, MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
        if (p == MAP_FAILED) {
            throw std::runtime_error("mmap failed");
        }
        map_ = static_cast<std::byte*>(p);
        std::byte* guard = map_ + map_bytes_ - page;
        ::mprotect(guard, page, PROT_NONE);
        data_ = guard - bytes.size();
        std::memcpy(data_, bytes.data(), bytes.size());
        size_ = bytes.size();
    }
    ~Guarded() { ::munmap(map_, map_bytes_); }
    Guarded(const Guarded&) = delete;
    Guarded& operator=(const Guarded&) = delete;
    std::span<const std::byte> span() const { return {data_, size_}; }

private:
    std::byte* map_ = nullptr;
    size_t map_bytes_ = 0;
    std::byte* data_ = nullptr;
    size_t size_ = 0;
};

// Every string the walk returns must lie inside STR and end there.
void expect_strings_inside(const ParsedMetaRecords& p, std::span<const std::byte> str) {
    const auto* lo = reinterpret_cast<const char*>(str.data());
    const auto* hi = lo + str.size();
    auto inside = [&](const char* s) {
        return s >= lo && s < hi && ::strnlen(s, hi - s) < static_cast<size_t>(hi - s);
    };
    for (const auto& e : p.entries) {
        EXPECT_TRUE(inside(e.meta.signature.data()));
        for (const FieldValue& v : e.meta.fields) {
            if (v.code == 's') {
                EXPECT_TRUE(inside(v.s));
            }
        }
    }
}

// ---- Valid sections ----------------------------------------------------------------------------------------------

TEST(ObservabilityEventMetaLoader, CPU_ReadsEveryRecordTheEmitterWrites) {
    Sections s;
    s.record(3, ZoneColorMeta{"red", 0xFF0000}, "a.cpp", 10);
    s.record(4, Nested{"nested", {0xA9A9A9, 2}, 0xBEEF}, "b.cpp", 20);
    s.record(
        5,
        AllWidths{
            "all",
            true,
            std::numeric_limits<int8_t>::min(),
            std::numeric_limits<uint16_t>::max(),
            -123456,
            std::numeric_limits<int64_t>::min(),
            Level::Low,
            kWideMax},
        "c.cpp",
        30);
    s.record(6, TwoStrings{"first", "second"});
    s.record(7, EventMeta{"plain"});
    s.record(8, Defaulted{"defaulted"});

    const ParsedMetaRecords p = s.parse();
    EXPECT_FALSE(p.stopped_early);
    ASSERT_EQ(p.entries.size(), 6u);
    expect_strings_inside(p, s.strs());

    EXPECT_EQ(p.entries[0].event_id, 3u);
    EXPECT_EQ(p.entries[0].name, "red");
    EXPECT_EQ(p.entries[0].file, "a.cpp");
    EXPECT_EQ(p.entries[0].line, 10u);
    const auto color = p.entries[0].meta.as<ZoneColorMeta>();
    ASSERT_TRUE(color.has_value());
    EXPECT_EQ(color->color, 0xFF0000u);

    const auto nested = p.entries[1].meta.as<Nested>();
    ASSERT_TRUE(nested.has_value());
    EXPECT_STREQ(nested->name, "nested");
    EXPECT_EQ(nested->style.color, 0xA9A9A9u);
    EXPECT_EQ(nested->style.verbosity, 2u);
    EXPECT_EQ(nested->flags, 0xBEEFu);

    const auto all = p.entries[2].meta.as<AllWidths>();
    ASSERT_TRUE(all.has_value());
    EXPECT_TRUE(all->on);
    EXPECT_EQ(all->i8, std::numeric_limits<int8_t>::min());
    EXPECT_EQ(all->u16, std::numeric_limits<uint16_t>::max());
    EXPECT_EQ(all->i32, -123456);
    EXPECT_EQ(all->i64, std::numeric_limits<int64_t>::min());
    EXPECT_EQ(all->level, Level::Low);
    EXPECT_EQ(all->wide, kWideMax);
    EXPECT_EQ(p.entries[2].line, 30u);

    // The name is the first string; a later one is just a field.
    EXPECT_EQ(p.entries[3].name, "first");
    const auto two = p.entries[3].meta.as<TwoStrings>();
    ASSERT_TRUE(two.has_value());
    EXPECT_STREQ(two->detail, "second");

    EXPECT_EQ(p.entries[4].name, "plain");
    EXPECT_TRUE(p.entries[4].meta.as<EventMeta>().has_value());

    // Members the event left to their defaults travel like any other value.
    const auto defaulted = p.entries[5].meta.as<Defaulted>();
    ASSERT_TRUE(defaulted.has_value());
    EXPECT_EQ(defaulted->color, 0x123456u);
    EXPECT_EQ(defaulted->verbosity, 4u);
}

TEST(ObservabilityEventMetaLoader, CPU_EmptyOrShortSectionHasNoRecords) {
    Sections s;
    ParsedMetaRecords p = s.parse();
    EXPECT_TRUE(p.entries.empty());
    EXPECT_FALSE(p.stopped_early);

    // Fewer bytes than a header after the last record are not a record.
    s.record(1, EventMeta{"x"});
    s.meta().resize(s.meta().size() + tt::observability_event::kRecordHeaderBytes - 1);
    p = s.parse();
    EXPECT_EQ(p.entries.size(), 1u);
    EXPECT_FALSE(p.stopped_early);
}

// ---- Damaged sections --------------------------------------------------------------------------------------------

TEST(ObservabilityEventMetaLoader, CPU_StopsAtARecordThatRunsPastTheSection) {
    Sections s;
    s.record(1, EventMeta{"kept"});
    const size_t first_end = s.meta().size();
    s.record(2, AllWidths{"cut", true, 1, 2, 3, 4, Level::High});
    const size_t full = s.meta().size();

    // Cut inside the second record's fields: the first survives and the walk stops.
    s.meta().resize(first_end + tt::observability_event::kRecordHeaderBytes + 6);
    ParsedMetaRecords p = s.parse();
    ASSERT_EQ(p.entries.size(), 1u);
    EXPECT_EQ(p.entries[0].name, "kept");
    EXPECT_TRUE(p.stopped_early);

    // Every cut point: never a record that does not fit, never a read past the cut, never a string outside the
    // section. The second record's last field (a uint64) ends the section, so only the full length holds both.
    Sections whole;
    whole.record(1, EventMeta{"kept"});
    whole.record(2, AllWidths{"cut", true, 1, 2, 3, 4, Level::High});
    const Guarded str(whole.strs());
    for (size_t len = 0; len <= full; len++) {
        const Guarded meta(std::span<const std::byte>(whole.meta()).first(len));
        p = parse_meta_records(meta.span(), str.span(), Sections::kStrVma);
        EXPECT_EQ(p.entries.size(), len < first_end ? 0u : len < full ? 1u : 2u) << "len " << len;
        expect_strings_inside(p, str.span());
    }
}

TEST(ObservabilityEventMetaLoader, CPU_StopsAtAnUnknownFieldCode) {
    Sections s;
    s.record(1, EventMeta{"kept"});
    s.header(2, "Bad:sz");
    s.put(s.str("bad"), 4);
    s.put(0, 4);
    const ParsedMetaRecords p = s.parse();
    ASSERT_EQ(p.entries.size(), 1u);
    EXPECT_EQ(p.entries[0].name, "kept");
    EXPECT_TRUE(p.stopped_early);
}

TEST(ObservabilityEventMetaLoader, CPU_StopsAtUnbalancedBraces) {
    for (std::string_view sig : {"Bad:s{w", "Bad:s}w{", "Bad:s{w}}", "Bad:s}"}) {
        Sections s;
        s.header(1, sig);
        s.put(s.str("bad"), 4);
        s.put(0, 4);
        const ParsedMetaRecords p = s.parse();
        EXPECT_TRUE(p.entries.empty()) << sig;
        EXPECT_TRUE(p.stopped_early) << sig;
    }
}

TEST(ObservabilityEventMetaLoader, CPU_StopsAtAPointerOutsideTheStrings) {
    // The signature pointer below the section, past its end, and at a string with no NUL before the end.
    for (uint32_t bad_sig : {0u, static_cast<uint32_t>(Sections::kStrVma - 1), 0xFFFFFFFFu}) {
        Sections s;
        s.str("unused");
        s.put(1, 4);
        s.put(bad_sig, 4);
        s.put(s.str("kernel.cpp"), 4);
        s.put(1, 4);
        const ParsedMetaRecords p = s.parse();
        EXPECT_TRUE(p.entries.empty()) << bad_sig;
        EXPECT_TRUE(p.stopped_early) << bad_sig;
    }
    {
        Sections s;
        s.record(1, EventMeta{"x"});
        s.strs().pop_back();  // the last string, the name, loses its NUL
        const ParsedMetaRecords p = s.parse();
        EXPECT_TRUE(p.entries.empty());
        EXPECT_TRUE(p.stopped_early);
    }
    {
        // A string field that points nowhere.
        Sections s;
        s.header(1, signature_of<EventMeta>());
        s.put(0x10, 4);
        const ParsedMetaRecords p = s.parse();
        EXPECT_TRUE(p.entries.empty());
        EXPECT_TRUE(p.stopped_early);
    }
}

TEST(ObservabilityEventMetaLoader, CPU_UnresolvableFileIsEmptyNotFatal) {
    Sections s;
    s.put(1, 4);
    s.put(s.str(signature_of<EventMeta>()), 4);
    s.put(0, 4);  // file
    s.put(9, 4);
    s.put(s.str("named"), 4);
    const ParsedMetaRecords p = s.parse();
    ASSERT_EQ(p.entries.size(), 1u);
    EXPECT_EQ(p.entries[0].name, "named");
    EXPECT_EQ(p.entries[0].file, "");
    EXPECT_FALSE(p.stopped_early);
}

// ---- Stale sections: the 16-byte {id, name, file, line} layout from before metadata -------------------------------

TEST(ObservabilityEventMetaLoader, CPU_StopsAtTheOldRecordLayout) {
    Sections s;
    s.old_record(1, "T1_Zone5");
    s.old_record(2, "T1_Zone6");
    const ParsedMetaRecords p = s.parse();
    EXPECT_TRUE(p.entries.empty());
    EXPECT_TRUE(p.stopped_early);
}

// An old name with a ':' reads as a signature and its record is mis-sized. What the walk does then is pinned here:
// it returns only unnamed records, which ingest_elf counts as malformed, so no zone gets a wrong name. That holds
// because the word after an old record is a small rebased id, never a string pointer; a version marker in the
// section would make it certain.
TEST(ObservabilityEventMetaLoader, CPU_OldNameWithAColonNamesNothing) {
    for (std::string_view name : {"a:w", "a:b", "a:h", "a:q", "a:s", "a:sw", "a:", "ns::x", "a:{w}"}) {
        Sections s;
        s.old_record(41, name);
        s.old_record(42, "T1_Zone6");
        s.old_record(43, "T1_Zone7");
        const ParsedMetaRecords p = s.parse();
        for (const auto& e : p.entries) {
            EXPECT_TRUE(e.name.empty()) << name << " named '" << e.name << "'";
        }
        expect_strings_inside(p, s.strs());
    }
}

// ---- Random damage -----------------------------------------------------------------------------------------------

TEST(ObservabilityEventMetaLoader, CPU_SurvivesRandomDamage) {
    Sections valid;
    valid.record(1, ZoneColorMeta{"red", 0xFF0000});
    valid.record(2, Nested{"nested", {1, 2}, 3});
    valid.record(3, AllWidths{"all", false, -1, 2, -3, -4, Level::High});
    valid.record(4, TwoStrings{"a", "b"});

    std::mt19937 rng(20261013);
    for (int iter = 0; iter < 20000; iter++) {
        std::vector<std::byte> meta = valid.meta();
        std::vector<std::byte> str = valid.strs();
        std::vector<std::byte>& target = rng() % 4 == 0 ? str : meta;
        const int flips = 1 + static_cast<int>(rng() % 8);
        for (int f = 0; f < flips && !target.empty(); f++) {
            target[rng() % target.size()] = static_cast<std::byte>(rng());
        }
        if (rng() % 3 == 0) {
            meta.resize(rng() % (meta.size() + 1));
        }
        if (rng() % 5 == 0) {
            str.resize(rng() % (str.size() + 1));
        }
        const Guarded guarded_meta(meta);
        const Guarded guarded_str(str);
        const ParsedMetaRecords p = parse_meta_records(guarded_meta.span(), guarded_str.span(), Sections::kStrVma);
        EXPECT_LE(p.entries.size(), meta.size() / tt::observability_event::kRecordHeaderBytes);
        expect_strings_inside(p, guarded_str.span());
        if (HasFailure()) {
            FAIL() << "iteration " << iter;
        }
    }
}

}  // namespace observability_event_loader_test
