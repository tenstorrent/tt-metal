// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <tt-metalium/experimental/device_observability.hpp>

#include <algorithm>
#include <array>
#include <bit>
#include <atomic>
#include <cstring>
#include <functional>
#include <memory_resource>
#include <optional>
#include <span>
#include <string>
#include <utility>
#include <variant>
#include <vector>

#include <tt_stl/assert.hpp>

#include "context/metal_context.hpp"
#include "hostdev/profiler_zone_id.h"
#include "hostdev/record_site.h"
#include "impl/streaming_profiler/decode.hpp"
#include "impl/streaming_profiler/service.hpp"
#include "impl/streaming_profiler/sync/clock_map.hpp"
#include "impl/streaming_profiler/external_receiver.hpp"
#include "llrt/call_sites.hpp"

namespace api = tt::tt_metal::experimental::device_observability;

namespace tt::tt_metal::streaming_profiler {

namespace {

using api::detail::code_bytes;
using api::detail::kTimestampCode;
constexpr uint16_t kBoolCode = api::detail::type_code<bool>();

// A field's size in a host record.
constexpr uint32_t host_bytes(uint16_t code) {
    return code == kTimestampCode ? sizeof(api::Timestamp) : code_bytes(code);
}

// How a Timestamp field's ticks come from what a packet of form `form` sent. The full form sends every Timestamp whole.
PayloadToRecord::Op timestamp_op(const tt::llrt::CallSite& site, uint32_t field, api::detail::TimestampWidth width) {
    using Op = PayloadToRecord::Op;
    return field == site.payload_publish_time_field                                    ? Op::Published
           : site.timestamps_as_deltas && width != api::detail::TimestampWidth::Bits64 ? Op::Delta
                                                                                       : Op::Time;
}

struct RecordTimes {
    const ClockMap& map;
    ClockMap::Reader& reader;
    uint32_t dev;
    uint64_t clock_offset;
    uint64_t published_ticks;
    // A host Timestamp holds the device time, then the host time.
    api::Timestamp timestamp(uint64_t ticks) const {
        const int64_t host = map.place_host(reader, dev, static_cast<int64_t>(ticks + clock_offset));
        return std::bit_cast<api::Timestamp>(std::array{ticks, static_cast<uint64_t>(host)});
    }
};

class PayloadFields final : public api::detail::FieldReader {
public:
    PayloadFields(
        const tt::llrt::CallSite& site,
        std::span<const std::byte> bytes,
        const tt::llrt::CallSite::PacketLayout& layout,
        std::span<const uint8_t> alternatives,
        std::pmr::memory_resource* arena,
        const RecordTimes* times) :
        FieldReader(alternatives, site.argument_field_counts, arena),
        site_(&site),
        times_(times),
        bytes_(bytes),
        codes_(site.payload_codes),
        layout_(&layout) {}

private:
    void read(std::span<std::byte> out) override {
        const size_t field = field_++;
        const uint16_t code = codes_[field];
        const tt::llrt::FieldBits bits = layout_->payload_bits[field];
        const uint64_t value = field_value(bytes_.data(), bits.bit, bits.width, code);
        if (code == kTimestampCode) {
            const PayloadToRecord::Op op = timestamp_op(*site_, field, layout_->timestamp_width);
            const api::Timestamp time = times_->timestamp(timestamp_ticks(op, times_->published_ticks, value));
            std::memcpy(out.data(), &time, out.size());
        } else {
            std::memcpy(out.data(), &value, out.size());
        }
    }

    const tt::llrt::CallSite* site_ = nullptr;
    const RecordTimes* times_ = nullptr;
    std::span<const std::byte> bytes_;
    std::span<const uint16_t> codes_;
    const tt::llrt::CallSite::PacketLayout* layout_ = nullptr;
    std::size_t field_ = 0;
};

// A call site's metadata, from the values its image held.
class MetadataFields final : public api::detail::FieldReader {
public:
    explicit MetadataFields(const tt::llrt::CallSite& site) : site_(&site) {}

private:
    void read(std::span<std::byte> out) override {
        std::visit([&](const auto& value) { std::memcpy(out.data(), &value, out.size()); }, site_->metadata[field_++]);
    }

    const tt::llrt::CallSite* site_ = nullptr;
    std::size_t field_ = 0;
};

// The alternative each of a call site's arguments takes in `record_type`, when the site sends records of that type: one
// argument per entry of record_type.arguments, or any number for a variadic type. A named type is matched by its name
// alone: one whose fields differ between the two sides has already stopped the program (tt::llrt::add_host_type).
std::optional<std::vector<uint8_t>> match(
    const tt::llrt::CallSite& site, const api::detail::RecordTypeDescriptor& record_type) {
    if (site.metadata_type != record_type.metadata_type ||
        (!record_type.variadic && site.arguments.size() != record_type.arguments.size())) {
        return std::nullopt;
    }
    std::vector<uint8_t> alternatives;
    for (std::size_t k = 0; k < site.arguments.size(); k++) {
        const tt::llrt::CallSite::Argument& argument = site.arguments[k];
        const std::span<const api::detail::ArgumentType> types = record_type.arguments[record_type.variadic ? 0 : k];
        const auto codes = std::span(site.payload_codes).subspan(argument.first_field, argument.field_bits.size());
        const auto fits = [&](const api::detail::ArgumentType& type) {
            return type.name == argument.type && type.array == argument.is_array &&
                   std::ranges::equal(type.codes, codes) &&
                   (type.count == 0 || type.count * argument.field_bits.size() == site.argument_field_counts[k]);
        };
        // An array of the exact length beats a span, as an exact match beats a conversion when a std::variant is built.
        auto chosen = std::ranges::find_if(
            types, [&](const api::detail::ArgumentType& type) { return type.count != 0 && fits(type); });
        if (chosen == types.end()) {
            chosen = std::ranges::find_if(types, fits);
        }
        if (chosen == types.end()) {
            return std::nullopt;
        }
        alternatives.push_back(uint8_t(chosen - types.begin()));
    }
    return alternatives;
}

// A record type's sample `sample`, which numbers its fields as the device's samples do (detail::sample_count).
class ProbeFields final : public api::detail::FieldReader {
public:
    ProbeFields(std::span<const uint16_t> codes, uint32_t sample) : codes_(codes), sample_(sample) {}

private:
    void read(std::span<std::byte> out) override {
        const uint16_t code = codes_[field_];
        const bool set = ((++field_ >> sample_) & 1) != 0;
        if (code == kBoolCode) {
            out[0] = std::byte(set);
        } else {
            std::ranges::fill(out, set ? std::byte{0xFF} : std::byte{0});
        }
    }

    std::span<const uint16_t> codes_;
    uint32_t sample_ = 0;
    size_t field_ = 0;
};

// The byte offset of each payload field in a record the decoder writes directly, from samples of the record. A payload
// without a fixed layout, or with a field narrower than its bytes, is built instead. A bool shows one bit, so it counts
// as whole when no other field shares its byte.
std::optional<std::vector<uint32_t>> direct_offsets(const api::detail::RecordTypeDescriptor& record_type) {
    const std::span<const uint16_t> codes = record_type.payload_codes;
    if (!record_type.fixed_layout) {
        return std::nullopt;
    }
    const uint32_t sample_count = api::detail::sample_count(uint32_t(codes.size()));
    std::vector<std::byte> samples(record_type.size * sample_count);
    for (uint32_t sample = 0; sample < sample_count; sample++) {
        ProbeFields fields(codes, sample);
        record_type.build(
            samples.data() + sample * record_type.size, api::detail::RecordOrigin{}, api::Timestamp{}, fields);
    }
    std::vector<tt::llrt::FieldBits> bits(codes.size());
    api::detail::field_bits_from_samples(samples.data(), sample_count, uint32_t(record_type.size), bits.data());
    std::vector<uint32_t> fields_in_byte(record_type.size);
    for (const tt::llrt::FieldBits& field : bits) {
        for (uint32_t byte = field.bit / 8; byte < (field.bit + field.width + 7) / 8; byte++) {
            fields_in_byte[byte]++;
        }
    }
    std::vector<uint32_t> offsets;
    for (size_t field = 0; field < codes.size(); field++) {
        const bool alone_in_byte = fields_in_byte[bits[field].bit / 8] == 1;
        const bool fills_its_bytes = bits[field].width == 8 * host_bytes(codes[field]) ||
                                     (codes[field] == kBoolCode && bits[field].width == 1 && alone_in_byte);
        if (bits[field].bit % 8 != 0 || !fills_its_bytes) {
            return std::nullopt;
        }
        offsets.push_back(bits[field].bit / 8);
    }
    return offsets;
}

PayloadToRecord payload_to_record(
    const tt::llrt::CallSite& site, const tt::llrt::CallSite::PacketLayout& layout, std::span<const uint32_t> offsets) {
    using Op = PayloadToRecord::Op;
    PayloadToRecord result;
    for (uint32_t field = 0; field < site.payload_codes.size(); field++) {
        const uint16_t code = site.payload_codes[field];
        const Op op = code == kTimestampCode ? timestamp_op(site, field, layout.timestamp_width) : Op::Copy;
        result.fields.push_back(
            {op, code, offsets[field], layout.payload_bits[field].bit, layout.payload_bits[field].width});
        if (op == Op::Delta || op == Op::Published) {
            result.follows_time.push_back(offsets[field]);
        }
    }
    return result;
}

// The layout spsc_short_records() writes a short form's records in, or none when the form's fields don't fit it.
// The payload is a distance, or the bytes of fields that share one qword of the record.
std::optional<profiler::SpscShortLayout> short_run_layout(const PayloadToRecord& payload_to_record, uint32_t stride) {
    using Layout = profiler::SpscShortLayout;
    Layout layout{.stride = stride};
    layout.payload_shuffle.fill(Layout::kZeroByte);
    std::optional<uint32_t> copied_qword;
    uint32_t filled_qwords = 0;
    const auto pick = [&](uint32_t qword, Layout::Qword from) {
        layout.payload_dwords[2 * qword] = 2 * from;
        layout.payload_dwords[2 * qword + 1] = 2 * from + 1;
        filled_qwords |= 1u << qword;
    };
    const auto qword_of = [](uint32_t byte) { return (byte - profiler::kSpscHeadBytes) / 8; };
    for (const PayloadToRecord::Field& field : payload_to_record.fields) {
        const uint32_t qword = qword_of(field.to);
        if (field.op == PayloadToRecord::Op::Copy) {
            const uint32_t bytes = code_bytes(field.code);
            if (layout.distance_mask != 0 || qword_of(field.to + bytes - 1) != qword ||
                copied_qword.value_or(qword) != qword) {
                return std::nullopt;
            }
            copied_qword = qword;
            pick(qword, Layout::kPayload);
            // The shuffle covers four records, two in each 128-bit half.
            for (uint32_t k = 0; k < bytes; k++) {
                for (uint32_t record = 0; record < layout.payload_shuffle.size() / 8; record++) {
                    layout.payload_shuffle[8 * record + field.to % 8 + k] =
                        uint8_t(8 * (record % 2) + Layout::kPayloadByte + field.bit / 8 + k);
                }
            }
        } else if (field.op == PayloadToRecord::Op::Delta) {
            if (copied_qword) {
                return std::nullopt;
            }
            layout.distance_mask = -1;
            pick(qword, Layout::kPayload);
            pick(qword + 1, Layout::kOffset);
        } else {  // Op::Published
            pick(qword, Layout::kTime);
            pick(qword + 1, Layout::kOffset);
        }
    }
    // A qword of the record's payload that no field fills would need zeros, which the run kernel doesn't write.
    if (filled_qwords != (1u << qword_of(layout.stride)) - 1) {
        return std::nullopt;
    }
    return layout;
}

constexpr uint64_t alternatives_key(uint32_t id, uint32_t type) { return uint64_t{id} << 32 | type; }

}  // namespace

RecordTypes::RecordTypes(std::vector<api::detail::RecordTypeDescriptor> record_types) :
    record_types_(std::move(record_types)) {
    for (size_t type = 0; type < record_types_.size(); type++) {
        std::optional<std::vector<uint32_t>> direct = direct_offsets(record_types_[type]);
        strides_[type] = direct ? static_cast<uint32_t>(record_types_[type].size) : profiler::kSpscPublishedBytes;
        if (direct) {
            direct_types_ |= uint64_t{1} << type;
            for (size_t field = 0; field < direct->size(); field++) {
                if (record_types_[type].payload_codes[field] == kTimestampCode) {
                    time_offsets_[type].push_back((*direct)[field]);
                }
            }
            field_offsets_[type] = std::move(*direct);
        }
    }
}

RecordRoutes::RecordRoutes(const RecordTypes& types, const api::detail::CallSites& call_sites) :
    types_(types), call_sites_(call_sites), routes_(TT_CALL_SITE_ID_COUNT), short_run_types_(TT_CALL_SITE_ID_COUNT) {}

RecordRoutes::Route RecordRoutes::resolve(uint32_t id) {
    // An image's call sites are registered when it loads, before it can run, so a record of an unregistered one means a
    // missing descriptor or a corrupt ring. Its packet forms are unknown, so the lane's time cursor could not be kept.
    const tt::llrt::CallSite* const site = call_sites_.find(id);
    TT_FATAL(site != nullptr, "streaming profiler: a record of call site {}, which no loaded image describes", id);
    Route resolved;
    for (uint32_t type = 0; type < types_.type_count(); type++) {
        if (std::optional<std::vector<uint8_t>> alternatives = match(*site, types_.record_type(type))) {
            resolved.types |= uint64_t{1} << type;
            alternatives_.emplace(alternatives_key(id, type), std::move(*alternatives));
        }
    }
    for (const tt::llrt::CallSite::PacketLayout& layout : site->packet_layouts) {
        if (layout.timestamp_width == api::detail::TimestampWidth::Bits16) {
            resolved.bits16_packet_words = uint16_t(layout.packet_words);
        } else if (layout.timestamp_width == api::detail::TimestampWidth::Bits64) {
            resolved.bits64_packet_words = uint16_t(layout.packet_words);
        }
    }
    if (const uint64_t direct = resolved.types & types_.direct_types_; direct != 0) {
        resolved.payload_to_record = payload_to_record_by_width(*site, types_.field_offsets_[std::countr_zero(direct)]);
    }
    set_route(id, resolved);
    return resolved;
}

uint64_t RecordRoutes::short_run_type(const Route& route) {
    const uint32_t type = std::countr_zero(route.types);
    if (route.bits16_packet_words != 2 || !std::has_single_bit(route.types) || types_.built(type) ||
        types_.stride(type) > profiler::SpscShortLayout::kMaxStride) {
        return 0;
    }
    const PayloadToRecord& payload_to_record =
        (*route.payload_to_record)[uint32_t(api::detail::TimestampWidth::Bits16)];
    const std::optional<profiler::SpscShortLayout> layout = short_run_layout(payload_to_record, types_.stride(type));
    if (!layout) {
        return 0;
    }
    if (!short_runs_[type]) {
        short_runs_[type] = {.layout = *layout, .payload_to_record = &payload_to_record};
    }
    return uint64_t{1} << type;
}

const PayloadToRecordByWidth* RecordRoutes::payload_to_record_by_width(
    const tt::llrt::CallSite& site, std::span<const uint32_t> field_offsets) {
    PayloadToRecordByWidth by_width;
    for (const tt::llrt::CallSite::PacketLayout& layout : site.packet_layouts) {
        by_width[uint32_t(layout.timestamp_width)] = payload_to_record(site, layout, field_offsets);
    }
    // The packet's payload is already the records' payload when the site has one form, a medium one as spsc_record()
    // reads it, and each field sits in it as in a record, past its head.
    const bool as_is =
        site.packet_layouts.size() == 1 &&
        site.packet_layouts[0].timestamp_width == api::detail::TimestampWidth::Bits32 &&
        std::ranges::all_of(
            by_width[uint32_t(site.packet_layouts[0].timestamp_width)].fields, [](const PayloadToRecord::Field& field) {
                return field.op == PayloadToRecord::Op::Copy && field.bit % 8 == 0 &&
                       field.width == 8 * code_bytes(field.code) &&
                       field.to == profiler::kSpscHeadBytes + field.bit / 8;
            });
    return as_is ? nullptr : &payload_to_record_by_width_.emplace_back(std::move(by_width));
}

std::span<const uint8_t> RecordRoutes::alternatives(uint32_t id, uint32_t type) const {
    return alternatives_.at(alternatives_key(id, type));
}

const std::byte* build_records(
    std::span<const PublishedRecord> published,
    uint32_t type,
    const RecordRoutes& routes,
    std::pmr::memory_resource& arena,
    const ClockMap& map,
    ClockMap::Reader& reader,
    uint32_t dev) {
    const api::detail::RecordTypeDescriptor& record_type = routes.types().record_type(type);
    auto* const out = static_cast<std::byte*>(arena.allocate(published.size() * record_type.size));
    for (size_t i = 0; i < published.size(); i++) {
        const PublishedRecord& record = published[i];
        const tt::llrt::CallSite& site = *routes.call_sites().find(record.origin.call_site_id);
        const RecordTimes times{map, reader, dev, record.clock_offset, record.ticks};
        const tt::llrt::CallSite::PacketLayout& layout = *std::ranges::find(
            site.packet_layouts, record.packet_words, &tt::llrt::CallSite::PacketLayout::packet_words);
        PayloadFields fields(
            site,
            std::span(record.bytes, record.byte_count),
            layout,
            routes.alternatives(record.origin.call_site_id, type),
            &arena,
            &times);
        record_type.build(out + i * record_type.size, record.origin, times.timestamp(record.ticks), fields);
    }
    return out;
}

}  // namespace tt::tt_metal::streaming_profiler

namespace tt::tt_metal::experimental::device_observability {

namespace internal = tt::tt_metal::streaming_profiler;

Callback detail::register_callback(
    std::string name,
    std::span<const RecordTypeDescriptor> record_types,
    std::function<void(const BatchData&)> callback) {
    static std::atomic<uint32_t> anonymous{0};
    if (name.empty()) {
        name = "callback-" + std::to_string(++anonymous);
    }
    // A process with callbacks and no MetalContext may be the external process, so it listens for Metal processes
    // until it creates a MetalContext of its own.
    bool metal = false;
    for (int context = 0; context < static_cast<int>(MAX_CONTEXT_COUNT); context++) {
        metal |= MetalContext::instance_exists(ContextId{context});
    }
    if (!metal) {
        internal::serve_metal_processes();
    }
    for (const RecordTypeDescriptor& record_type : record_types) {
        tt::llrt::add_host_type(record_type.metadata_type, record_type.metadata_codes);
        for (const std::span<const ArgumentType> alternatives : record_type.arguments) {
            for (const ArgumentType& type : alternatives) {
                tt::llrt::add_host_type(type.name, type.codes);
            }
        }
    }
    return Callback(internal::service().add_consumer(
        std::move(name),
        std::vector<RecordTypeDescriptor>(record_types.begin(), record_types.end()),
        std::move(callback)));
}

void detail::visit_chunks(
    const BatchData& batch, const std::function<void(std::size_t type, std::size_t from, std::size_t count)>& visit) {
    if (batch.counts.size() == 1) {
        visit(0, 0, batch.counts[0]);
        return;
    }
    std::array<uint32_t, internal::kMaxTypes> visited{};
    uint32_t type = 0, count = 0;
    const auto visit_chunk = [&] {
        if (count != 0) {
            visit(type, visited[type], count);
            visited[type] += count;
            count = 0;
        }
    };
    for (const uint32_t record_type : batch.order) {
        if (record_type != type) {
            visit_chunk();
            type = record_type;
        }
        count++;
    }
    visit_chunk();
}

const void* detail::build_site_metadata(
    const RecordOrigin& origin, const void* (*build)(FieldReader& fields), void (*destroy)(const void* metadata)) {
    return origin.lane->call_sites->build_metadata(
        origin.call_site_id,
        [&](const tt::llrt::CallSite& site) {
            internal::MetadataFields fields(site);
            return build(fields);
        },
        destroy);
}

const SourceLocation& detail::location_of(const CallSites& call_sites, uint32_t call_site_id) {
    return call_sites.find(call_site_id)->location;
}

const detail::CaptureLanes& detail::CaptureLanes::create(
    const internal::CaptureContext& capture, std::shared_ptr<const CallSites> call_sites) {
    auto* lanes = new CaptureLanes;
    lanes->call_sites_ = std::move(call_sites);
    for (const internal::CaptureContext::Device& device : capture.devices) {
        std::vector<RecordLane>& by_lane = lanes->by_device_.emplace_back();
        for (const Core& core : device.lanes) {
            by_lane.push_back(
                {.core = core,
                 .call_sites = lanes->call_sites_.get(),
                 .metadata = lanes->call_sites_->metadata(),
                 .lanes = lanes});
        }
    }
    return *lanes;
}

void detail::retain(const CaptureLanes& lanes) { lanes.references_.fetch_add(1, std::memory_order_relaxed); }

void detail::release(const CaptureLanes& lanes) {
    if (lanes.references_.fetch_sub(1, std::memory_order_acq_rel) == 1) {
        delete &lanes;
    }
}

Callback::Callback(Callback&& other) noexcept : id_(std::exchange(other.id_, detail::CallbackId{})) {}

Callback& Callback::operator=(Callback&& other) noexcept {
    if (this != &other) {
        reset();
        id_ = std::exchange(other.id_, detail::CallbackId{});
    }
    return *this;
}

Callback::~Callback() { reset(); }

void Callback::reset() noexcept {
    if (id_ != detail::CallbackId{}) {
        internal::service().remove_consumer(std::exchange(id_, detail::CallbackId{}));
    }
}

bool IsActive() { return internal::service().is_active(); }

constinit thread_local Timestamp::TscToSteadyLine Timestamp::tsc_to_steady_line_{};

std::chrono::steady_clock::time_point Timestamp::refill_tsc_to_steady_line(int64_t tsc) {
    const internal::SteadyClock& steady = internal::service().steady();
    const std::optional<internal::Segment<int64_t>> segment = steady.segment_at(tsc);
    TT_FATAL(
        segment, "streaming profiler: a TSC tick maps to steady_clock only after a capture's first host sync burst");
    tsc_to_steady_line_ = {
        .from = segment->from,
        .to = segment->to,
        .origin = segment->origin,
        .base_ns = steady.base_ns(),
        .value = segment->value,
        .slope = segment->slope};
    // Past the newest SteadyClock sample, segment_at() returns an empty segment that only extrapolates, so calling
    // time() here would just refill again.
    return tsc_to_steady_line_.at(tsc);
}

}  // namespace tt::tt_metal::experimental::device_observability
