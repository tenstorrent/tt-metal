// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <tt-metalium/experimental/streaming_profiler.hpp>

#include <algorithm>
#include <array>
#include <bit>
#include <atomic>
#include <cstring>
#include <functional>
#include <memory_resource>
#include <mutex>
#include <optional>
#include <ranges>
#include <span>
#include <string>
#include <utility>
#include <vector>

#include <tt_stl/assert.hpp>

#include "hostdev/profiler_zone_id.h"
#include "hostdev/record_site.h"
#include "impl/streaming_profiler/decode.hpp"
#include "impl/streaming_profiler/service.hpp"
#include "impl/streaming_profiler/sync/clock_map.hpp"
#include "llrt/call_sites.hpp"

namespace api = tt::tt_metal::experimental::streaming_profiler;

namespace tt::tt_metal::streaming_profiler {

namespace {

using api::detail::code_bytes;
using api::detail::kStringCode;
using api::detail::kTimestampCode;
constexpr uint16_t kBoolCode = api::detail::type_code<bool>();

// A field's size in a host record.
constexpr uint32_t host_bytes(uint16_t code) {
    return code == kTimestampCode ? sizeof(api::Timestamp) : code_bytes(code);
}

// How a Timestamp field's ticks come from what its record sent.
DirectForm::Op timestamp_op(const tt::llrt::CallSite& site, uint32_t field) {
    using Op = DirectForm::Op;
    return field == site.published_field ? Op::Published : site.timestamps_as_distances ? Op::Distance : Op::Time;
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

class SiteFields final : public api::detail::FieldReader {
public:
    SiteFields(
        const tt::llrt::CallSite& site,
        std::span<const std::byte> bytes,
        std::span<const uint16_t> codes,
        std::span<const tt::llrt::FieldBits> bits,
        std::span<const uint8_t> alternatives = {},
        std::pmr::memory_resource* arena = nullptr,
        const RecordTimes* times = nullptr) :
        FieldReader(alternatives, site.value_fields, arena),
        site_(&site),
        times_(times),
        bytes_(bytes),
        codes_(codes),
        bits_(bits) {}

private:
    void read(void* out) override {
        const size_t field = field_++;
        const uint16_t code = codes_[field];
        const uint64_t value = field_value(bytes_.data(), bits_[field].bit, bits_[field].width, code);
        if (code == kStringCode) {
            *static_cast<std::string_view*>(out) = site_->strings->string_at(uint32_t(value));
        } else if (code == kTimestampCode) {
            const DirectForm::Op op = timestamp_op(*site_, field);
            *static_cast<api::Timestamp*>(out) = times_->timestamp(timestamp_ticks(op, times_->published_ticks, value));
        } else {
            std::memcpy(out, &value, code_bytes(code));
        }
    }

    const tt::llrt::CallSite* site_ = nullptr;
    const RecordTimes* times_ = nullptr;
    std::span<const std::byte> bytes_;
    std::span<const uint16_t> codes_;
    std::span<const tt::llrt::FieldBits> bits_;
    std::size_t field_ = 0;
};

std::optional<std::vector<uint8_t>> value_alternatives(
    const tt::llrt::CallSite& site, std::span<const api::detail::ValueType> types) {
    std::vector<uint8_t> alternatives;
    for (std::size_t k = 0; k < site.values.size(); k++) {
        const tt::llrt::CallSite::Value& value = site.values[k];
        const auto codes = std::span(site.data_codes).subspan(value.first_field, value.field_bits.size());
        const auto fits = [&](const api::detail::ValueType& type) {
            return type.name == value.type && type.array == value.is_array && std::ranges::equal(type.codes, codes) &&
                   (type.count == 0 || type.count * value.field_bits.size() == site.value_fields[k]);
        };
        // An array of the exact length beats a span, as an exact match beats a conversion when a std::variant is built.
        auto chosen = std::ranges::find_if(
            types, [&](const api::detail::ValueType& type) { return type.count != 0 && fits(type); });
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

// The alternative each of a call site's values takes in `kind`, when the site sends records of that kind. A named type
// is matched by its name alone: one whose fields differ between the two sides has already stopped the program
// (tt::llrt::add_host_type).
std::optional<std::vector<uint8_t>> match(const tt::llrt::CallSite& site, const api::detail::Kind& kind) {
    if (!kind.metadata_type.empty() && site.metadata_type != kind.metadata_type) {
        return std::nullopt;
    }
    if (!kind.value_types.empty()) {
        return site.values.empty() ? std::nullopt : value_alternatives(site, kind.value_types);
    }
    const bool payload_matches = kind.data_type.empty() ? site.values.empty()
                                                        : site.values.size() == 1 && !site.values[0].is_array &&
                                                              site.values[0].type == kind.data_type;
    return payload_matches ? std::optional(std::vector<uint8_t>()) : std::nullopt;
}

class ProbeFields final : public api::detail::FieldReader {
public:
    ProbeFields(std::span<const uint16_t> codes, size_t probed_field) : codes_(codes), probed_field_(probed_field) {}

private:
    void read(void* out) override {
        const uint16_t code = codes_[field_];
        const bool probed = field_++ == probed_field_;
        if (code == kBoolCode) {
            *static_cast<bool*>(out) = probed;
        } else {
            std::memset(out, probed ? 0xFF : 0, host_bytes(code));
        }
    }

    std::span<const uint16_t> codes_;
    size_t probed_field_ = 0, field_ = 0;
};

// The byte offset of each data field in a record the decoder writes directly, found by building one record per field
// with only that field's bits set. A type with values or a String field, or with a field narrower than its bytes, is
// built instead. A bool shows one bit, so it counts as whole when no other field shares its byte.
std::optional<std::vector<uint32_t>> direct_offsets(const api::detail::Kind& kind) {
    const std::span<const uint16_t> codes = kind.data_codes;
    if (!kind.value_types.empty() || std::ranges::find(codes, kStringCode) != codes.end()) {
        return std::nullopt;
    }
    std::vector<std::byte> samples(kind.size * codes.size());
    for (size_t field = 0; field < codes.size(); field++) {
        ProbeFields fields(codes, field);
        kind.build(samples.data() + field * kind.size, api::detail::Source{}, api::Timestamp{}, fields);
    }
    std::vector<uint32_t> offsets;
    std::vector<tt::llrt::FieldBits> bits;
    for (size_t field = 0; field < codes.size(); field++) {
        bits.push_back(tt::llrt::find_set_bits(std::span(samples).subspan(field * kind.size, kind.size)));
    }
    const auto alone_in_byte = [&](size_t field) {
        const uint32_t byte = bits[field].bit / 8;
        return std::ranges::none_of(std::views::iota(size_t{0}, codes.size()), [&](size_t other) {
            return other != field && bits[other].bit / 8 <= byte &&
                   byte < (bits[other].bit + bits[other].width + 7) / 8;
        });
    };
    for (size_t field = 0; field < codes.size(); field++) {
        const bool fills_its_bytes = bits[field].width == 8 * host_bytes(codes[field]) ||
                                     (codes[field] == kBoolCode && bits[field].width == 1 && alone_in_byte(field));
        if (bits[field].bit % 8 != 0 || !fills_its_bytes) {
            return std::nullopt;
        }
        offsets.push_back(bits[field].bit / 8);
    }
    return offsets;
}

DirectForm direct_form(
    const tt::llrt::CallSite& site, std::span<const tt::llrt::FieldBits> bits, std::span<const uint32_t> offsets) {
    using Op = DirectForm::Op;
    DirectForm form;
    for (uint32_t field = 0; field < site.data_codes.size(); field++) {
        const uint16_t code = site.data_codes[field];
        const Op op = code == kTimestampCode ? timestamp_op(site, field) : Op::Copy;
        form.fields.push_back({op, code, offsets[field], bits[field].bit, bits[field].width});
        if (op == Op::Distance || op == Op::Published) {
            form.follows_time.push_back(offsets[field]);
        }
    }
    return form;
}

// The layout spsc_short_records() writes a short form's records in, or none when the form's fields don't fit it.
// The payload is a distance, or the bytes of fields that share one qword of the record.
std::optional<profiler::SpscShortLayout> short_run_layout(const DirectForm& form, uint32_t stride) {
    using Layout = profiler::SpscShortLayout;
    Layout layout{.stride = stride};
    layout.payload_shuffle.fill(Layout::kZeroByte);
    std::optional<uint32_t> copied_qword;
    uint32_t filled_qwords = 0;
    const auto pick = [&](uint32_t qword, Layout::Qword from) {
        layout.data_dwords[2 * qword] = 2 * from;
        layout.data_dwords[2 * qword + 1] = 2 * from + 1;
        filled_qwords |= 1u << qword;
    };
    const auto qword_of = [](uint32_t byte) { return (byte - profiler::kSpscHeadBytes) / 8; };
    for (const DirectForm::Field& field : form.fields) {
        const uint32_t qword = qword_of(field.to);
        if (field.op == DirectForm::Op::Copy) {
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
        } else if (field.op == DirectForm::Op::Distance) {
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
    // A qword of the record's data that no field fills would need zeros, which the run kernel doesn't write.
    if (filled_qwords != (1u << qword_of(layout.stride)) - 1) {
        return std::nullopt;
    }
    return layout;
}

constexpr uint64_t alternatives_key(uint32_t id, uint32_t type) { return uint64_t{id} << 32 | type; }

}  // namespace

RecordRoutes::RecordRoutes(std::vector<api::detail::Kind> kinds) :
    kinds_(std::move(kinds)), routes_(TT_ZONE_ID_COUNT), short_run_types_(TT_ZONE_ID_COUNT) {
    for (size_t type = 0; type < kinds_.size(); type++) {
        std::optional<std::vector<uint32_t>> direct = direct_offsets(kinds_[type]);
        strides_[type] = direct ? static_cast<uint32_t>(kinds_[type].size) : profiler::kSpscPublishedBytes;
        if (direct) {
            direct_types_ |= uint64_t{1} << type;
            for (size_t field = 0; field < direct->size(); field++) {
                if (kinds_[type].data_codes[field] == kTimestampCode) {
                    time_offsets_[type].push_back((*direct)[field]);
                }
            }
            field_offsets_[type] = std::move(*direct);
        }
    }
}

RecordRoutes::Route RecordRoutes::resolve(uint32_t id) {
    Route resolved;
    if (const tt::llrt::CallSite* site = tt::llrt::find_call_site(id)) {
        for (uint32_t type = 0; type < kinds_.size(); type++) {
            if (std::optional<std::vector<uint8_t>> alternatives = match(*site, kinds_[type])) {
                resolved.types |= uint64_t{1} << type;
                alternatives_.emplace(alternatives_key(id, type), std::move(*alternatives));
            }
        }
        for (const tt::llrt::CallSite::Form& form : site->forms) {
            if (form.time_form == api::detail::TimeForm::Short) {
                resolved.short_length = uint8_t(form.body_words);
            } else if (form.time_form == api::detail::TimeForm::Full) {
                resolved.full_length = uint8_t(form.body_words);
            }
        }
        if ((resolved.types & direct_types_) != 0) {
            resolved.forms = direct_forms(*site, field_offsets_[std::countr_zero(resolved.types & direct_types_)]);
        }
    }
    set_route(id, resolved);
    return resolved;
}

uint64_t RecordRoutes::short_run_type(const Route& route) {
    const uint32_t type = std::countr_zero(route.types);
    if (route.short_length != 1 || !std::has_single_bit(route.types) || built(type) ||
        strides_[type] > profiler::SpscShortLayout::kMaxStride) {
        return 0;
    }
    const DirectForm& form = (*route.forms)[uint32_t(api::detail::TimeForm::Short)];
    const std::optional<profiler::SpscShortLayout> layout = short_run_layout(form, strides_[type]);
    if (!layout) {
        return 0;
    }
    if (!short_runs_[type]) {
        short_runs_[type] = {.layout = *layout, .form = &form};
    }
    return uint64_t{1} << type;
}

const DirectForms* RecordRoutes::direct_forms(const tt::llrt::CallSite& site, std::span<const uint32_t> field_offsets) {
    DirectForms forms;
    for (const tt::llrt::CallSite::Form& form : site.forms) {
        forms[uint32_t(form.time_form)] = direct_form(site, form.data_bits, field_offsets);
    }
    // The payload is already the records' data when the site has one form and each field sits in it as in a record,
    // past its head.
    const bool as_is =
        site.forms.size() == 1 &&
        std::ranges::all_of(forms[uint32_t(site.forms[0].time_form)].fields, [](const DirectForm::Field& field) {
            return field.op == DirectForm::Op::Copy && field.bit % 8 == 0 &&
                   field.width == 8 * code_bytes(field.code) && field.to == profiler::kSpscHeadBytes + field.bit / 8;
        });
    return as_is ? nullptr : &forms_.emplace_back(std::move(forms));
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
    const api::detail::Kind& kind = routes.kind(type);
    auto* const out = static_cast<std::byte*>(arena.allocate(published.size() * kind.size));
    for (size_t i = 0; i < published.size(); i++) {
        const PublishedRecord& record = published[i];
        const tt::llrt::CallSite& site = *tt::llrt::find_call_site(record.source.zone_id);
        const RecordTimes times{map, reader, dev, record.clock_offset, record.ticks};
        const tt::llrt::CallSite::Form& form =
            *std::ranges::find(site.forms, record.length, &tt::llrt::CallSite::Form::body_words);
        SiteFields fields(
            site,
            std::span(record.bytes, record.byte_count),
            site.data_codes,
            form.data_bits,
            routes.alternatives(record.source.zone_id, type),
            &arena,
            &times);
        kind.build(out + i * kind.size, record.source, times.timestamp(record.ticks), fields);
    }
    return out;
}

}  // namespace tt::tt_metal::streaming_profiler

namespace tt::tt_metal::experimental::streaming_profiler {

namespace internal = tt::tt_metal::streaming_profiler;

Callback detail::register_callback(
    std::string name, std::span<const Kind> kinds, std::function<void(const BatchData&)> callback) {
    static std::atomic<uint32_t> anonymous{0};
    if (name.empty()) {
        name = "callback-" + std::to_string(++anonymous);
    }
    for (const Kind& kind : kinds) {
        tt::llrt::add_host_type(kind.data_type, kind.data_codes);
        tt::llrt::add_host_type(kind.metadata_type, kind.metadata_codes);
        for (const ValueType& type : kind.value_types) {
            tt::llrt::add_host_type(type.name, type.codes);
        }
    }
    return Callback(internal::service().add_consumer(
        std::move(name), std::vector<Kind>(kinds.begin(), kinds.end()), std::move(callback)));
}

void detail::visit_stretches(
    const BatchData& batch, const std::function<void(std::size_t type, std::size_t from, std::size_t count)>& visit) {
    if (batch.counts.size() == 1) {
        visit(0, 0, batch.counts[0]);
        return;
    }
    std::array<uint32_t, internal::RecordRoutes::kMaxTypes> visited{};
    uint32_t type = 0, count = 0;
    const auto visit_stretch = [&] {
        if (count != 0) {
            visit(type, visited[type], count);
            visited[type] += count;
            count = 0;
        }
    };
    for (const uint32_t record_type : batch.order) {
        if (record_type != type) {
            visit_stretch();
            type = record_type;
        }
        count++;
    }
    visit_stretch();
}

std::atomic<const void*> detail::site_metadata[TT_ZONE_ID_COUNT];

const void* detail::build_site_metadata(uint32_t id, const void* (*build)(FieldReader& fields)) {
    static std::mutex building;
    const std::lock_guard lock(building);
    const void* metadata = site_metadata[id].load(std::memory_order_relaxed);
    if (metadata == nullptr) {
        const tt::llrt::CallSite& site = *tt::llrt::find_call_site(id);
        internal::SiteFields fields(site, site.metadata, site.metadata_codes, site.metadata_bits);
        metadata = build(fields);
        site_metadata[id].store(metadata, std::memory_order_release);
    }
    return metadata;
}

const SourceLocation& detail::location_of(uint32_t zone_id) { return tt::llrt::find_call_site(zone_id)->location; }

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

}  // namespace tt::tt_metal::experimental::streaming_profiler
