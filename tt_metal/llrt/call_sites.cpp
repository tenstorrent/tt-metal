// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "call_sites.hpp"

#include <algorithm>
#include <atomic>
#include <cstring>
#include <deque>
#include <map>
#include <memory>
#include <mutex>
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

namespace api = tt::tt_metal::experimental::device_observability;

// Kernels and the host show the same struct by name, so every showing of a name must have the same fields. A long,
// unsigned long or size_t field is 32 bits on the device and 64 on the host, which neither compiler can tell from a
// fixed-width type, so it shows as a difference here.
void check_same_fields(
    std::string_view name, std::span<const uint16_t> device_codes, std::span<const uint16_t> host_codes) {
    if (!std::ranges::equal(device_codes, host_codes)) {
        TT_THROW(
            "streaming profiler: {} has different fields in kernels than on the host; rebuild the host, and use "
            "fixed-width types instead of long or size_t",
            name);
    }
}

// A struct's fields as a kernel shows it: each one's type code and bits.
struct TypeLayout {
    std::vector<uint16_t> codes;
    std::vector<FieldBits> bits;
};

// The field codes of each type a host callback reads, by name, which the process's callbacks all share.
struct HostTypes {
    std::mutex mutex;
    std::map<std::string, std::vector<uint16_t>, std::less<>> codes;
};
HostTypes& host_types() {
    static ttsl::Indestructible<HostTypes> instance;
    return instance.get();
}
// Checks a type's fields against those of the host type a callback reads under its name, if any.
void check_host_fields(std::string_view name, std::span<const uint16_t> codes) {
    HostTypes& host = host_types();
    const std::lock_guard lock(host.mutex);
    if (const auto found = host.codes.find(name); found != host.codes.end()) {
        check_same_fields(name, codes, found->second);
    }
}

// The stall call site, kernel_profiler::write_pending_stall(), which every image shares and none describes, since it
// writes its records without a DevicePublish call. They are ppfmt's stall record, which the producer checks against
// DevicePublish's layout of it, of the types the host's Stall takes.
CallSite make_stall_site() {
    using ppfmt = kernel_profiler::ppfmt;
    constexpr auto codes = api::detail::leaf_type_codes<api::Interval>;
    // An Interval's Timestamps lie back to back, which the producer checks too.
    constexpr uint32_t kTimestampBits = 8 * api::detail::code_bytes(api::detail::kTimestampCode);
    // The start travels as a distance back from the end, which is the record's time and isn't sent.
    std::vector<FieldBits> payload_bits(codes.size(), {0, ppfmt::STALL_START_DELTA_BITS});
    payload_bits[ppfmt::STALL_PUBLISH_TIME_FIELD] = {};
    return {
        .metadata_type = api::detail::type_name<api::StallMetadata>,
        .payload_codes = {codes.begin(), codes.end()},
        .arguments =
            {{.type = api::detail::type_name<api::Interval>,
              .first_field = 0,
              .field_bits = {{0, kTimestampBits}, {kTimestampBits, kTimestampBits}}}},
        .argument_field_counts = {uint32_t(codes.size())},
        .payload_publish_time_field = ppfmt::STALL_PUBLISH_TIME_FIELD,
        .packet_layouts =
            {{.timestamp_width = api::detail::TimestampWidth::Bits32,
              .packet_words = ppfmt::STALL_PACKET_WORDS,
              .payload_bits = std::move(payload_bits)}},
        .timestamps_as_deltas = true};
}

// Stores `text` for the life of the process and returns a view of the stored copy, the same one for the same text. Each
// string sits in a node of its own, which inserting others never moves, so the views stay valid.
std::string_view store_string(std::string_view text) {
    static ttsl::Indestructible<std::pair<std::mutex, std::set<std::string, std::less<>>>> store;
    auto& [mutex, texts] = store.get();
    const std::lock_guard lock(mutex);
    if (const auto stored = texts.find(text); stored != texts.end()) {
        return *stored;
    }
    return *texts.emplace(text).first;
}

// This process's images' share of its call sites: the next id an image gets, and which ids send to the external
// process.
struct ProcessImages {
    std::mutex mutex;
    uint32_t next_id = 0;
    // The ids of the images' call sites whose records go to the external process, in load order.
    std::vector<uint32_t> ids_sending_to_external_process;
    std::mutex listener_mutex;  // held through each call, so removing the listener waits out the calls in flight
    std::function<void()> listener;
};

ProcessImages& process_images() {
    static ttsl::Indestructible<ProcessImages> instance;
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

// Each field of struct `type`, from its field bits at `field_bits_at`. A field fits its type's width, one as wide as
// its type starts on a byte, and no two fields share a bit.
std::vector<FieldBits> read_field_bits(
    const Section& contents, uint64_t field_bits_at, std::span<const uint16_t> codes, std::string_view type) {
    std::vector<FieldBits> bits = contents.read_table<FieldBits>(field_bits_at, codes.size());
    for (size_t field = 0; field < codes.size(); field++) {
        const uint32_t type_bits = 8 * api::detail::code_bytes(codes[field]);
        TT_FATAL(
            bits[field].width < type_bits || (bits[field].width == type_bits && bits[field].bit % 8 == 0),
            "streaming profiler: field {} of {} takes bits {} to {}, which its {}-bit type can't",
            field,
            type,
            bits[field].bit,
            bits[field].bit + bits[field].width,
            type_bits);
    }
    std::vector<FieldBits> by_bit = bits;
    std::ranges::sort(by_bit, {}, &FieldBits::bit);
    for (size_t k = 1; k < by_bit.size(); k++) {
        TT_FATAL(
            by_bit[k - 1].bit + by_bit[k - 1].width <= by_bit[k].bit,
            "streaming profiler: two fields of {} share bit {}",
            type,
            by_bit[k].bit);
    }
    return bits;
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
            values.emplace_back(strings.of(value));
        } else {
            values.emplace_back(value);
        }
    }
    return values;
}

// Where a field of the slice's argument lies in the payload, or width 0 when the slice doesn't hold it. A form sends a
// Timestamp's distance in the slice that starts where the Timestamp does.
FieldBits find_in_payload(const api::detail::ArgumentSlice& slice, FieldBits field, bool as_delta) {
    const bool holds = as_delta
                           ? 8u * slice.from == field.bit
                           : 8u * slice.from <= field.bit && field.bit + field.width <= 8u * (slice.from + slice.bytes);
    if (!holds) {
        return {};
    }
    return {field.bit - 8u * slice.from + 8u * slice.to, as_delta ? 8u * slice.bytes : field.width};
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
            argument.field_bits != 0 ? read_field_bits(contents, argument.field_bits, codes, strings.at(argument.type))
                                     : std::vector{FieldBits{0, 8 * api::detail::code_bytes(codes[0])}};
        for (uint32_t element = 0; element < argument.count; element++) {
            for (const FieldBits& field : bits) {
                fields.push_back(
                    {uint32_t(site.arguments.size()), {field.bit + 8 * element * argument.size, field.width}});
            }
        }
        site.arguments.push_back(
            {.type = strings.at(argument.type),
             .is_enum = argument.type != 0 && argument.field_bits == 0,
             .first_field = first_field,
             .is_array = argument.is_array != 0,
             .field_bits = std::move(bits)});
        site.argument_field_counts.push_back(argument.field_count * argument.count);
        first_field += argument.field_count * argument.count;
    }
    return fields;
}

// Each payload field's bits in the payload of form `form`'s packets, which the form's slices give. The device lays
// every field into one slice, by the field bits the host reads, so only a kernel built from other DevicePublish
// headers leaves one out. The field a record is published into isn't sent.
std::vector<FieldBits> find_payload_bits(
    const CallSite& site,
    std::span<const ArgumentField> fields,
    api::detail::TimestampWidth width,
    std::span<const api::detail::ArgumentSlice> slices) {
    // The slice holding each byte of each argument. One argument's slices don't overlap, so the slice holding a field's
    // first byte is the only one that can hold the field.
    std::vector<std::vector<int32_t>> slice_at(site.arguments.size());
    for (size_t k = 0; k < slices.size(); k++) {
        std::vector<int32_t>& at = slice_at[slices[k].argument];
        at.resize(std::max<size_t>(at.size(), slices[k].from + slices[k].bytes), -1);
        std::fill_n(at.begin() + slices[k].from, slices[k].bytes, int32_t(k));
    }
    std::vector<FieldBits> bits;
    for (uint32_t field = 0; field < fields.size(); field++) {
        const auto [argument, bits_in_argument] = fields[field];
        const bool as_delta = site.timestamps_as_deltas && width != api::detail::TimestampWidth::Bits64 &&
                              site.payload_codes[field] == api::detail::kTimestampCode;
        const std::vector<int32_t>& at = slice_at[argument];
        const uint32_t first_byte = bits_in_argument.bit / 8;
        const int32_t slice = first_byte < at.size() ? at[first_byte] : -1;
        bits.push_back(slice < 0 ? FieldBits{} : find_in_payload(slices[slice], bits_in_argument, as_delta));
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

// Adds an image's call sites to the process's, by the ids its load gave them. Their strings point into the image, which
// CallSites::add stores before the image goes.
void add_call_sites(const std::string& elf_path, const ll_api::ElfFile& elf) {
    const Section descriptors(elf, ".device_publish_descriptors"), contents(elf, ".device_publish_descriptor_contents"),
        string_entries(elf, ".device_publish_strings");
    const ImageStrings strings(contents, string_entries, elf_path);
    for (const api::detail::CallSiteDescriptor& descriptor : descriptors.read_table<api::detail::CallSiteDescriptor>(
             descriptors.vma, descriptors.bytes.size() / sizeof(api::detail::CallSiteDescriptor))) {
        CallSite site{
            .location =
                {.file = strings.of(descriptor.file),
                 .function = strings.at(descriptor.function),
                 .line = descriptor.line},
            .metadata_type = strings.at(descriptor.metadata_type),
            .metadata_codes = contents.read_table<uint16_t>(descriptor.metadata_codes, descriptor.metadata_field_count),
            .payload_codes = contents.read_table<uint16_t>(descriptor.payload_codes, descriptor.payload_field_count),
            .payload_publish_time_field = descriptor.payload_publish_time_field,
            .sends_to_external_process = descriptor.sends_to_external_process != 0};
        site.metadata_bits =
            read_field_bits(contents, descriptor.metadata_field_bits, site.metadata_codes, site.metadata_type);
        site.metadata =
            read_metadata(site, contents.read_table<std::byte>(descriptor.metadata, descriptor.metadata_size), strings);
        const std::vector<ArgumentField> fields = read_arguments(
            site,
            contents.read_table<api::detail::CallArgument>(
                descriptor.payload_arguments, descriptor.payload_argument_count),
            contents,
            strings);
        // Only a payload whose Timestamps travel as distances has a full form, unless its records go to the external
        // process too, which then take the full form only and so send no distances.
        const auto layouts =
            contents.read_table<api::detail::PacketLayout>(descriptor.packet_layouts, descriptor.packet_layout_count);
        site.timestamps_as_deltas =
            std::ranges::find(
                layouts, api::detail::TimestampWidth::Bits64, &api::detail::PacketLayout::timestamp_width) !=
            layouts.end();
        for (const api::detail::PacketLayout& layout : layouts) {
            site.packet_layouts.push_back(
                {.timestamp_width = layout.timestamp_width,
                 .packet_words = layout.packet_words,
                 .payload_bits = find_payload_bits(
                     site,
                     fields,
                     layout.timestamp_width,
                     contents.read_table<api::detail::ArgumentSlice>(layout.slices, layout.slice_count))});
        }
        const bool sends_to_external_process = site.sends_to_external_process;
        process_call_sites().add(descriptor.id, std::move(site));
        if (sends_to_external_process) {
            ProcessImages& images = process_images();
            const std::lock_guard lock(images.mutex);
            images.ids_sending_to_external_process.push_back(descriptor.id);
        }
    }
}

}  // namespace

void load_call_sites(const std::string& elf_path, ll_api::ElfFile& elf) {
    uint64_t ids_vma = 0;
    const size_t count = elf.GetSectionContents(".device_publish_ids", ids_vma).size();
    if (count == 0) {
        return;
    }

    ProcessImages& images = process_images();
    uint32_t base = 0;
    {
        const std::lock_guard lock(images.mutex);
        TT_FATAL(
            uint64_t{images.next_id} + count <= TT_STALL_CALL_SITE_ID,
            "streaming profiler: '{}' has {} call sites, and only {} of the process's {} call-site ids remain",
            elf_path,
            count,
            TT_STALL_CALL_SITE_ID - images.next_id,
            TT_STALL_CALL_SITE_ID);
        base = std::exchange(images.next_id, images.next_id + static_cast<uint32_t>(count));
    }
    elf.RebaseCallSiteIds(base);

    add_call_sites(elf_path, elf);
    log_debug(tt::LogLLRuntime, "streaming profiler: '{}' -> call site ids [{}, {})", elf_path, base, base + count);
    const std::lock_guard listening(images.listener_mutex);
    if (images.listener) {
        images.listener();
    }
}

void set_call_site_listener(std::function<void()> listener) {
    ProcessImages& images = process_images();
    const std::lock_guard listening(images.listener_mutex);
    images.listener = std::move(listener);
}

std::vector<uint32_t> ids_of_call_sites_sending_to_external_process(size_t first) {
    ProcessImages& images = process_images();
    const std::lock_guard lock(images.mutex);
    const std::vector<uint32_t>& ids = images.ids_sending_to_external_process;
    return std::vector<uint32_t>(ids.begin() + first, ids.end());
}

CallSites& process_call_sites() {
    static ttsl::Indestructible<CallSites> instance;
    return instance.get();
}

void add_host_type(std::string_view name, std::span<const uint16_t> codes) {
    {
        HostTypes& host = host_types();
        const std::lock_guard lock(host.mutex);
        host.codes.try_emplace(std::string(name), codes.begin(), codes.end());
    }
    process_call_sites().check_host_type(name, codes);
}

}  // namespace tt::llrt

namespace tt::tt_metal::experimental::device_observability::detail {

namespace llrt = tt::llrt;

struct CallSites::State {
    std::mutex mutex;
    std::deque<llrt::CallSite> sites;
    // The fields of each struct name the sites show, which every showing of the name must match.
    std::map<std::string, llrt::TypeLayout, std::less<>> types;
    // The metadata built for the sites, each with what deletes it.
    std::mutex building;
    std::vector<std::unique_ptr<const void, void (*)(const void*)>> built;

    void add_type(std::string_view name, std::span<const uint16_t> codes, std::span<const llrt::FieldBits> bits) {
        const auto [shown, added] = types.try_emplace(std::string(name));
        if (added) {
            shown->second = {{codes.begin(), codes.end()}, {bits.begin(), bits.end()}};
        } else if (!std::ranges::equal(shown->second.codes, codes) || !std::ranges::equal(shown->second.bits, bits)) {
            TT_THROW("streaming profiler: two kernels send different types named {}; give each its own name", name);
        }
        llrt::check_host_fields(name, codes);
    }
};

CallSites::CallSites() :
    state_(std::make_unique<State>()),
    by_id_(std::make_unique<std::atomic<const llrt::CallSite*>[]>(TT_CALL_SITE_ID_COUNT)),
    metadata_(std::make_unique<std::atomic<const void*>[]>(TT_CALL_SITE_ID_COUNT)) {
    by_id_[TT_STALL_CALL_SITE_ID].store(
        &state_->sites.emplace_back(llrt::make_stall_site()), std::memory_order_release);
}

CallSites::~CallSites() = default;

void CallSites::add(uint32_t id, llrt::CallSite site) {
    site.location.file = llrt::store_string(site.location.file);
    site.location.function = llrt::store_string(site.location.function);
    site.metadata_type = llrt::store_string(site.metadata_type);
    for (llrt::CallSite::Argument& argument : site.arguments) {
        argument.type = llrt::store_string(argument.type);
    }
    for (auto& value : site.metadata) {
        if (auto* text = std::get_if<std::string_view>(&value)) {
            *text = llrt::store_string(*text);
        }
    }
    const std::lock_guard lock(state_->mutex);
    state_->add_type(site.metadata_type, site.metadata_codes, site.metadata_bits);
    for (const llrt::CallSite::Argument& argument : site.arguments) {
        if (!argument.type.empty()) {
            state_->add_type(
                argument.type,
                std::span(site.payload_codes).subspan(argument.first_field, argument.field_bits.size()),
                argument.field_bits);
        }
    }
    by_id_[id].store(&state_->sites.emplace_back(std::move(site)), std::memory_order_release);
}

const void* CallSites::build_metadata(
    uint32_t id,
    const std::function<const void*(const llrt::CallSite& site)>& build,
    void (*destroy)(const void* metadata)) const {
    const std::lock_guard lock(state_->building);
    const void* metadata = metadata_[id].load(std::memory_order_relaxed);
    if (metadata == nullptr) {
        metadata = build(*find(id));
        state_->built.emplace_back(metadata, destroy);
        metadata_[id].store(metadata, std::memory_order_release);
    }
    return metadata;
}

void CallSites::check_host_type(std::string_view name, std::span<const uint16_t> codes) const {
    const std::lock_guard lock(state_->mutex);
    if (const auto shown = state_->types.find(name); shown != state_->types.end()) {
        llrt::check_same_fields(name, shown->second.codes, codes);
    }
}

}  // namespace tt::tt_metal::experimental::device_observability::detail
