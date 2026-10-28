// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "call_sites.hpp"

#include <algorithm>
#include <array>
#include <atomic>
#include <bit>
#include <cstring>
#include <deque>
#include <map>
#include <mutex>
#include <optional>
#include <set>
#include <string>
#include <unordered_map>
#include <utility>
#include <variant>
#include <vector>

#include <tt-logger/tt-logger.hpp>
#include <tt_stl/assert.hpp>
#include <tt_stl/indestructible.hpp>

#include "hostdev/profiler_zone_id.h"
#include "hostdev/streaming_profiler_common.h"
#include "tt_elffile.hpp"

namespace tt::llrt {

namespace {

namespace api = tt::tt_metal::experimental::streaming_profiler;

// The fields each struct name has, as kernels and the host show it. The host matches a type by its name, so every
// showing of a name must have the same fields. A long, unsigned long or size_t field is 32 bits on the device and 64 on
// the host, which neither compiler can tell from a fixed-width type, so it shows as a difference here.
class TypeLayouts {
public:
    void add_device_type(std::string_view name, std::span<const uint16_t> codes, std::span<const FieldBits> bits) {
        const auto [shown, added] = device_.try_emplace(std::string(name));
        if (added) {
            shown->second = {{codes.begin(), codes.end()}, {bits.begin(), bits.end()}};
        } else if (!std::ranges::equal(shown->second.codes, codes) || !std::ranges::equal(shown->second.bits, bits)) {
            TT_THROW("streaming profiler: two kernels send different types named {}; give each its own name", name);
        }
        if (const auto host = host_.find(name); host != host_.end()) {
            check_same_fields(name, host->second, codes);
        }
    }
    void add_host_type(std::string_view name, std::span<const uint16_t> codes) {
        host_.try_emplace(std::string(name), codes.begin(), codes.end());
        if (const auto device = device_.find(name); device != device_.end()) {
            check_same_fields(name, device->second.codes, codes);
        }
    }

private:
    struct Layout {
        std::vector<uint16_t> codes;
        std::vector<FieldBits> bits;
    };
    static void check_same_fields(
        std::string_view name, std::span<const uint16_t> device_codes, std::span<const uint16_t> host_codes) {
        if (!std::ranges::equal(device_codes, host_codes)) {
            TT_THROW(
                "streaming profiler: {} has different fields in kernels than on the host; rebuild the host, and use "
                "fixed-width types instead of long or size_t",
                name);
        }
    }

    std::map<std::string, Layout, std::less<>> device_;
    std::map<std::string, std::vector<uint16_t>, std::less<>> host_;
};

// The stall call site, which kernel_profiler::write_pending_stall() writes without an entry in any ELF: a zone named
// STALL_ZONE_NAME whose records all take the medium form, its end the record's time and its start a 32-bit distance
// before its end.
CallSite make_stall_site() {
    return {
        .metadata_type = api::detail::type_name<api::ZoneMetadata>,
        .payload_codes = {api::detail::kTimestampCode, api::detail::kTimestampCode},
        .metadata_codes = {api::detail::kStringCode},
        .metadata_bits = {{0, 64}},
        .arguments =
            {{.type = api::detail::type_name<api::Interval>, .first_field = 0, .field_bits = {{0, 64}, {64, 64}}}},
        .argument_field_counts = {2},
        .packet_layouts =
            {{.time_form = api::detail::TimeForm::Medium,
              .body_words = kernel_profiler::ppfmt::STALL_BODY_WORDS,
              .payload_bits = {{0, 32}, {0, 0}}}},
        .timestamps_as_distances = true,
        .payload_publish_time_field = 1,
        .metadata = {api::STALL_ZONE_NAME}};
}

struct Registry {
    Registry() { by_id[TT_ZONE_STALL_ID].store(&sites.emplace_back(make_stall_site()), std::memory_order_release); }

    std::mutex mutex;
    uint32_t next_id = 0;
    std::deque<CallSite> sites;  // a record may hold its site's address for the life of the process
    std::array<std::atomic<const CallSite*>, TT_ZONE_ID_COUNT> by_id{};
    TypeLayouts types;
};

Registry& call_site_registry() {
    static ttsl::Indestructible<Registry> instance;
    return instance.get();
}

struct Section {
    Section(const ll_api::ElfFile& elf, std::string_view name) { bytes = elf.GetSectionContents(name, vma); }

    template <typename T>
    std::vector<T> read_table(uint64_t address, uint64_t count) const {
        std::vector<T> out(count);
        if (count != 0) {
            TT_FATAL(
                address >= vma && address - vma + count * sizeof(T) <= bytes.size(),
                "streaming profiler: a call site's table lies outside its section");
            std::memcpy(out.data(), bytes.data() + (address - vma), count * sizeof(T));
        }
        return out;
    }

    uint64_t vma = 0;
    std::span<const std::byte> bytes;
};

// Each field of struct `type`, of `struct_size` bytes, from its samples at `samples_at`, one per field with only that
// field set. A field fits its type's width, one as wide as its type starts on a byte, and no two fields share a bit.
std::vector<FieldBits> read_field_bits(
    const Section& contents,
    uint64_t samples_at,
    uint32_t struct_size,
    std::span<const uint16_t> codes,
    std::string_view type) {
    const std::vector<std::byte> samples =
        contents.read_table<std::byte>(samples_at, uint64_t{struct_size} * codes.size());
    std::vector<FieldBits> bits;
    for (size_t field = 0; field < codes.size(); field++) {
        const FieldBits found = find_set_bits(std::span(samples).subspan(field * struct_size, struct_size));
        const uint32_t type_bits = 8 * api::detail::code_bytes(codes[field]);
        TT_FATAL(
            found.width < type_bits || (found.width == type_bits && found.bit % 8 == 0),
            "streaming profiler: field {} of {} takes bits {} to {} of its sample, which its {}-bit type can't",
            field,
            type,
            found.bit,
            found.bit + found.width,
            type_bits);
        bits.push_back(found);
    }
    std::vector<FieldBits> by_bit = bits;
    std::ranges::sort(by_bit, {}, &FieldBits::bit);
    for (size_t k = 1; k < by_bit.size(); k++) {
        TT_FATAL(
            by_bit[k - 1].bit + by_bit[k - 1].width <= by_bit[k].bit,
            "streaming profiler: two fields of {} share bit {} in their samples",
            type,
            by_bit[k].bit);
    }
    return bits;
}

// A text a call site names, kept for the life of the process.
std::string_view keep(std::string_view text) {
    static ttsl::Indestructible<std::pair<std::mutex, std::set<std::string, std::less<>>>> kept;
    auto& [mutex, texts] = kept.get();
    const std::lock_guard lock(mutex);
    return *texts.emplace(text).first;
}

// An image's texts: those in its .device_publish_descriptor_contents at a link address, and its Strings by hash, from
// its .device_publish_strings. Each translation unit has its own entry for a text it uses, so a text repeats; two
// different texts with one hash fail the load, as does a String the image has no text for.
class ImageStrings {
public:
    ImageStrings(const Section& contents, const Section& string_entries, const std::string& elf_path) :
        contents_(contents), elf_path_(elf_path) {
        for (const api::detail::StringEntry& entry : string_entries.read_table<api::detail::StringEntry>(
                 string_entries.vma, string_entries.bytes.size() / sizeof(api::detail::StringEntry))) {
            const std::string_view text = at(entry.text).substr(0, entry.size);
            const auto [held, added] = by_hash_.try_emplace(entry.hash, text);
            TT_FATAL(
                added || held->second == text,
                "streaming profiler: '{}' has two strings with one hash, \"{}\" and \"{}\"; change either text",
                elf_path,
                held->second,
                text);
        }
    }
    // The text at `address`, up to its NUL.
    std::string_view at(uint32_t address) const {
        const uint64_t offset = uint64_t{address} - contents_.vma;
        if (offset >= contents_.bytes.size()) {
            return {};
        }
        const std::string_view rest(
            reinterpret_cast<const char*>(contents_.bytes.data()) + offset, contents_.bytes.size() - offset);
        return rest.substr(0, rest.find('\0'));
    }
    std::string_view of(uint64_t hash) const {
        const auto found = by_hash_.find(hash);
        TT_FATAL(found != by_hash_.end(), "streaming profiler: '{}' names a string it has no text for", elf_path_);
        return found->second;
    }

private:
    const Section& contents_;
    const std::string& elf_path_;
    std::unordered_map<uint64_t, std::string_view> by_hash_;
};

// The metadata's fields, read out of the struct as the device compiler laid it out, each String as its text.
std::vector<std::variant<uint64_t, std::string_view>> read_metadata(
    const CallSite& site, std::span<const std::byte> metadata, const ImageStrings& strings) {
    std::vector<std::variant<uint64_t, std::string_view>> values;
    for (size_t field = 0; field < site.metadata_codes.size(); field++) {
        const uint16_t code = site.metadata_codes[field];
        const FieldBits bits = site.metadata_bits[field];
        const uint64_t value = api::detail::field_value(metadata.data(), bits.bit, bits.width, code);
        if (code == api::detail::kStringCode) {
            values.emplace_back(keep(strings.of(value)));
        } else {
            values.emplace_back(value);
        }
    }
    return values;
}

// A form sends a Timestamp's distance in the slice that starts where the Timestamp does.
FieldBits find_in_payload(
    std::span<const api::detail::ArgumentSlice> slices, uint32_t argument, FieldBits field, bool as_distance) {
    for (const api::detail::ArgumentSlice& slice : slices) {
        const bool holds =
            as_distance ? 8u * slice.from == field.bit
                        : 8u * slice.from <= field.bit && field.bit + field.width <= 8u * (slice.from + slice.bytes);
        if (slice.argument == argument && holds) {
            return {field.bit - 8u * slice.from + 8u * slice.to, as_distance ? 8u * slice.bytes : field.width};
        }
    }
    return {};
}

// A payload field: the argument holding it, and its bits in that argument.
struct ArgumentField {
    uint32_t argument;
    FieldBits bits;
};

// Adds the call site's arguments, and returns each payload field where its argument holds it, an array argument's
// fields once per element.
std::vector<ArgumentField> read_arguments(
    CallSite& site,
    std::span<const api::detail::CallArgument> arguments,
    const Section& contents,
    const ImageStrings& strings) {
    std::vector<ArgumentField> fields;
    uint32_t first_field = 0;
    for (const api::detail::CallArgument& argument : arguments) {
        const auto codes = std::span(site.payload_codes).subspan(first_field, argument.field_count);
        std::vector<FieldBits> bits =
            argument.samples != 0
                ? read_field_bits(contents, argument.samples, argument.size, codes, strings.at(argument.type))
                : std::vector{FieldBits{0, 8 * api::detail::code_bytes(codes[0])}};
        for (uint32_t element = 0; element < argument.count; element++) {
            for (const FieldBits& field : bits) {
                fields.push_back(
                    {uint32_t(site.arguments.size()), {field.bit + 8 * element * argument.size, field.width}});
            }
        }
        site.arguments.push_back(
            {.type = keep(strings.at(argument.type)),
             .first_field = first_field,
             .is_array = argument.is_array != 0,
             .field_bits = std::move(bits)});
        site.argument_field_counts.push_back(argument.field_count * argument.count);
        first_field += argument.field_count * argument.count;
    }
    return fields;
}

// Each payload field's bits in the payload of a form's packets, which the form's slices give. The device lays every
// field into one slice, from the same samples the host reads (internal/debug/device_publish_encoding.h), so only a
// mismatch between the two leaves one out. The field a record is published into isn't sent.
std::vector<FieldBits> find_payload_bits(
    const CallSite& site, std::span<const ArgumentField> fields, std::span<const api::detail::ArgumentSlice> slices) {
    std::vector<FieldBits> bits;
    for (uint32_t field = 0; field < fields.size(); field++) {
        const auto [argument, bits_in_argument] = fields[field];
        const bool as_distance =
            site.timestamps_as_distances && site.payload_codes[field] == api::detail::kTimestampCode;
        bits.push_back(find_in_payload(slices, argument, bits_in_argument, as_distance));
        TT_FATAL(
            bits.back().width != 0 || field == site.payload_publish_time_field,
            "streaming profiler: {}:{}: field {} of {} lies outside the slices its records carry; the kernel's "
            "DevicePublish header and the host disagree about its layout, so rebuild both",
            site.location.file,
            site.location.line,
            (field - site.arguments[argument].first_field) % site.arguments[argument].field_bits.size(),
            site.arguments[argument].type.empty() ? "an argument" : site.arguments[argument].type);
    }
    return bits;
}

std::vector<std::pair<uint32_t, CallSite>> read_call_sites(const std::string& elf_path, const ll_api::ElfFile& elf) {
    const Section descriptors(elf, ".device_publish_descriptors"), contents(elf, ".device_publish_descriptor_contents"),
        string_entries(elf, ".device_publish_strings");
    const ImageStrings strings(contents, string_entries, elf_path);
    std::vector<std::pair<uint32_t, CallSite>> sites;
    for (const api::detail::CallSiteDescriptor& descriptor : descriptors.read_table<api::detail::CallSiteDescriptor>(
             descriptors.vma, descriptors.bytes.size() / sizeof(api::detail::CallSiteDescriptor))) {
        CallSite site{
            .location =
                {.file = keep(strings.of(descriptor.file)),
                 .function = keep(strings.at(descriptor.function)),
                 .line = descriptor.line},
            .metadata_type = keep(strings.at(descriptor.metadata_type)),
            .payload_codes = contents.read_table<uint16_t>(descriptor.payload_codes, descriptor.payload_field_count),
            .metadata_codes = contents.read_table<uint16_t>(descriptor.metadata_codes, descriptor.metadata_field_count),
            .payload_publish_time_field = descriptor.payload_publish_time_field};
        site.metadata_bits = read_field_bits(
            contents, descriptor.metadata_samples, descriptor.metadata_size, site.metadata_codes, site.metadata_type);
        site.metadata =
            read_metadata(site, contents.read_table<std::byte>(descriptor.metadata, descriptor.metadata_size), strings);
        const std::vector<ArgumentField> fields = read_arguments(
            site,
            contents.read_table<api::detail::CallArgument>(
                descriptor.payload_arguments, descriptor.payload_argument_count),
            contents,
            strings);
        // Only data whose Timestamps travel as distances has a full form.
        const auto layouts =
            contents.read_table<api::detail::PacketLayout>(descriptor.packet_layouts, descriptor.packet_layout_count);
        site.timestamps_as_distances =
            std::ranges::find(layouts, api::detail::TimeForm::Full, &api::detail::PacketLayout::time_form) !=
            layouts.end();
        for (const api::detail::PacketLayout& layout : layouts) {
            site.packet_layouts.push_back(
                {.time_form = layout.time_form,
                 .body_words = layout.body_words,
                 .payload_bits = find_payload_bits(
                     site,
                     fields,
                     contents.read_table<api::detail::ArgumentSlice>(layout.slices, layout.slice_count))});
        }
        sites.emplace_back(descriptor.id, std::move(site));
    }
    return sites;
}

}  // namespace

FieldBits find_set_bits(std::span<const std::byte> sample) {
    std::optional<uint32_t> lowest;
    uint32_t highest = 0;
    for (size_t byte = 0; byte < sample.size(); byte++) {
        const auto set = std::to_integer<uint8_t>(sample[byte]);
        if (set != 0) {
            lowest = lowest.value_or(uint32_t(8 * byte) + std::countr_zero(set));
            highest = uint32_t(8 * byte) + 7 - std::countl_zero(set);
        }
    }
    return lowest ? FieldBits{*lowest, highest + 1 - *lowest} : FieldBits{};
}

void load_call_sites(const std::string& elf_path, ll_api::ElfFile& elf) {
    uint64_t ids_vma = 0;
    const size_t count = elf.GetSectionContents(".device_publish_ids", ids_vma).size();
    if (count == 0) {
        return;
    }

    Registry& registry = call_site_registry();
    uint32_t base = 0;
    {
        const std::lock_guard lock(registry.mutex);
        TT_FATAL(
            uint64_t{registry.next_id} + count <= TT_ZONE_STALL_ID,
            "streaming profiler: '{}' has {} call sites, and only {} of the process's {} call-site ids remain",
            elf_path,
            count,
            TT_ZONE_STALL_ID - registry.next_id,
            TT_ZONE_STALL_ID);
        base = std::exchange(registry.next_id, registry.next_id + static_cast<uint32_t>(count));
    }
    elf.RebaseCallSiteIds(base);

    std::vector<std::pair<uint32_t, CallSite>> sites = read_call_sites(elf_path, elf);
    const std::lock_guard lock(registry.mutex);
    for (auto& [id, site] : sites) {
        registry.types.add_device_type(site.metadata_type, site.metadata_codes, site.metadata_bits);
        for (const CallSite::Argument& argument : site.arguments) {
            if (!argument.type.empty()) {
                registry.types.add_device_type(
                    argument.type,
                    std::span(site.payload_codes).subspan(argument.first_field, argument.field_bits.size()),
                    argument.field_bits);
            }
        }
        registry.by_id[id].store(&registry.sites.emplace_back(std::move(site)), std::memory_order_release);
    }
    log_debug(tt::LogLLRuntime, "streaming profiler: '{}' -> call site ids [{}, {})", elf_path, base, base + count);
}

const CallSite* find_call_site(uint32_t id) { return call_site_registry().by_id[id].load(std::memory_order_acquire); }

void add_host_type(std::string_view name, std::span<const uint16_t> codes) {
    Registry& registry = call_site_registry();
    const std::lock_guard lock(registry.mutex);
    registry.types.add_host_type(name, codes);
}

}  // namespace tt::llrt
