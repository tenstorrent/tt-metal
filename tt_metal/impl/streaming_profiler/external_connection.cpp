// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "impl/streaming_profiler/external_connection.hpp"

#include <fcntl.h>
#include <poll.h>
#include <sys/eventfd.h>
#include <sys/file.h>
#include <sys/inotify.h>
#include <sys/socket.h>
#include <sys/stat.h>
#include <sys/time.h>
#include <sys/un.h>
#include <unistd.h>

#include <algorithm>
#include <array>
#include <bit>
#include <cerrno>
#include <cstddef>
#include <cstring>
#include <mutex>
#include <span>
#include <string>
#include <string_view>
#include <utility>
#include <variant>
#include <vector>

#include <tt-logger/tt-logger.hpp>
#include <tt_stl/assert.hpp>

#include "hostdev/streaming_profiler_common.h"
#include "impl/streaming_profiler/spsc_marker_decode.hpp"
#include "protobuf/external_process.pb.h"
#include "llrt/call_sites.hpp"

namespace tt::tt_metal::streaming_profiler::external {

namespace {

namespace api = tt::tt_metal::experimental::device_observability;

// Each message is a 32-bit byte count, then a serialized Envelope.
// The longest message the external process takes; a longer one ends the connection.
constexpr uint32_t kMaxMessageBytes = 64u << 20;
// How long the external process waits for a Metal process that connected to send its hello.
constexpr timeval kDescriptorTimeout{.tv_sec = 5, .tv_usec = 0};

bool send_all(int fd, std::span<const std::byte> data) {
    while (!data.empty()) {
        const ssize_t n = ::send(fd, data.data(), data.size(), MSG_NOSIGNAL);
        if (n <= 0) {
            if (n < 0 && errno == EINTR) {
                continue;
            }
            return false;
        }
        data = data.subspan(size_t(n));
    }
    return true;
}
bool recv_all(int fd, std::span<std::byte> data) {
    while (!data.empty()) {
        const ssize_t n = ::recv(fd, data.data(), data.size(), 0);
        if (n <= 0) {
            if (n < 0 && errno == EINTR) {
                continue;
            }
            return false;
        }
        data = data.subspan(size_t(n));
    }
    return true;
}
bool send_message(int fd, const std::vector<uint8_t>& message) {
    const auto bytes = uint32_t(message.size());
    return send_all(fd, std::as_bytes(std::span(&bytes, 1))) && send_all(fd, std::as_bytes(std::span(message)));
}
std::optional<std::vector<uint8_t>> recv_message(int fd) {
    uint32_t bytes = 0;
    if (!recv_all(fd, std::as_writable_bytes(std::span(&bytes, 1))) || bytes > kMaxMessageBytes) {
        return std::nullopt;
    }
    std::vector<uint8_t> message(bytes);
    if (!recv_all(fd, std::as_writable_bytes(std::span(message)))) {
        return std::nullopt;
    }
    return message;
}

std::vector<uint8_t> serialize(const proto::Envelope& envelope) {
    std::vector<uint8_t> bytes(envelope.ByteSizeLong());
    TT_FATAL(
        envelope.SerializeToArray(bytes.data(), static_cast<int>(bytes.size())),
        "external connection: a message didn't serialize");
    return bytes;
}

constexpr uint16_t code(char kind, uint32_t bits) { return api::detail::field_code(kind, bits); }

proto::FieldType field_type(uint16_t field_code) {
    switch (field_code) {
        case code('b', 8): return proto::FIELD_TYPE_BOOL;
        case code('c', 8): return proto::FIELD_TYPE_CHAR;
        case code('u', 8): return proto::FIELD_TYPE_UINT8;
        case code('u', 16): return proto::FIELD_TYPE_UINT16;
        case code('u', 32): return proto::FIELD_TYPE_UINT32;
        case code('u', 64): return proto::FIELD_TYPE_UINT64;
        case code('i', 8): return proto::FIELD_TYPE_INT8;
        case code('i', 16): return proto::FIELD_TYPE_INT16;
        case code('i', 32): return proto::FIELD_TYPE_INT32;
        case code('i', 64): return proto::FIELD_TYPE_INT64;
        case code('f', 32): return proto::FIELD_TYPE_FLOAT32;
        case code('f', 64): return proto::FIELD_TYPE_FLOAT64;
        case code('s', 64): return proto::FIELD_TYPE_STRING;
        case code('t', 64): return proto::FIELD_TYPE_TIMESTAMP;
        default: TT_THROW("external connection: no field type has code {:#x}", field_code);
    }
}
uint16_t field_code(proto::FieldType type) {
    switch (type) {
        case proto::FIELD_TYPE_BOOL: return code('b', 8);
        case proto::FIELD_TYPE_CHAR: return code('c', 8);
        case proto::FIELD_TYPE_UINT8: return code('u', 8);
        case proto::FIELD_TYPE_UINT16: return code('u', 16);
        case proto::FIELD_TYPE_UINT32: return code('u', 32);
        case proto::FIELD_TYPE_UINT64: return code('u', 64);
        case proto::FIELD_TYPE_INT8: return code('i', 8);
        case proto::FIELD_TYPE_INT16: return code('i', 16);
        case proto::FIELD_TYPE_INT32: return code('i', 32);
        case proto::FIELD_TYPE_INT64: return code('i', 64);
        case proto::FIELD_TYPE_FLOAT32: return code('f', 32);
        case proto::FIELD_TYPE_FLOAT64: return code('f', 64);
        case proto::FIELD_TYPE_STRING: return code('s', 64);
        case proto::FIELD_TYPE_TIMESTAMP: return code('t', 64);
        default: TT_THROW("external connection: unknown field type {}", static_cast<int>(type));
    }
}

// The call site with id `id` as the protocol describes it. It sends only full records, so it has one packet layout.
void describe_call_site(uint32_t id, const tt::llrt::CallSite& site, proto::CallSite& out) {
    TT_FATAL(
        site.packet_layouts.size() == 1 &&
            site.packet_layouts[0].timestamp_width == api::detail::TimestampWidth::Bits64,
        "streaming profiler: call site {} sends to the external process in other than the full form",
        id);
    out.set_id(id);
    out.mutable_location()->set_file(std::string(site.location.file));
    out.mutable_location()->set_function(std::string(site.location.function));
    out.mutable_location()->set_line(site.location.line);
    proto::Metadata& metadata = *out.mutable_metadata();
    metadata.set_type_name(std::string(site.metadata_type));
    for (size_t f = 0; f < site.metadata.size(); f++) {
        proto::MetadataField& field = *metadata.add_fields();
        field.set_type(field_type(site.metadata_codes[f]));
        if (const auto* text = std::get_if<std::string_view>(&site.metadata[f])) {
            field.set_text(std::string(*text));
        } else {
            field.set_bits(std::get<uint64_t>(site.metadata[f]));
        }
    }
    const std::vector<tt::llrt::FieldBits>& payload_bits = site.packet_layouts[0].payload_bits;
    for (size_t a = 0; a < site.arguments.size(); a++) {
        const tt::llrt::CallSite::Argument& argument = site.arguments[a];
        const uint32_t fields = site.argument_field_counts[a];
        proto::PayloadValue& value = *out.add_values();
        if (argument.type.empty()) {
            value.set_scalar(field_type(site.payload_codes[argument.first_field]));
        } else if (argument.is_enum) {
            value.set_enum_name(std::string(argument.type));
        } else {
            value.set_struct_name(std::string(argument.type));
        }
        if (argument.is_array) {
            value.set_array_length(fields / static_cast<uint32_t>(argument.field_bits.size()));
        }
        for (uint32_t f = argument.first_field; f < argument.first_field + fields; f++) {
            proto::PayloadField& field = *value.add_fields();
            field.set_type(field_type(site.payload_codes[f]));
            field.set_bit_offset(payload_bits[f].bit);
            field.set_bit_size(payload_bits[f].width);
        }
    }
}

// The call site a message describes, for the external process to add. Its strings point into the message, which
// CallSites::add stores. A full record's packet is its first word, its time's two words, then the payload up to its
// last field, which a record may extend with words no field uses.
tt::llrt::CallSite read_call_site(const proto::CallSite& message) {
    tt::llrt::CallSite site;
    site.location = {
        .file = message.location().file(),
        .function = message.location().function(),
        .line = message.location().line()};
    site.metadata_type = message.metadata().type_name();
    for (const proto::MetadataField& field : message.metadata().fields()) {
        site.metadata_codes.push_back(field_code(field.type()));
        if (field.has_text()) {
            site.metadata.emplace_back(std::string_view(field.text()));
        } else {
            site.metadata.emplace_back(uint64_t{field.bits()});
        }
    }
    tt::llrt::CallSite::PacketLayout layout{.timestamp_width = api::detail::TimestampWidth::Bits64};
    uint32_t payload_end_bits = 0;
    for (const proto::PayloadValue& value : message.values()) {
        const auto fields = static_cast<uint32_t>(value.fields_size());
        const uint32_t elements = value.has_array_length() ? value.array_length() : 1;
        TT_FATAL(
            elements != 0 && fields % elements == 0,
            "external connection: call site {} has a value of {} fields in {} elements",
            message.id(),
            fields,
            elements);
        tt::llrt::CallSite::Argument& argument = site.arguments.emplace_back();
        argument.type = value.has_enum_name() ? value.enum_name() : value.struct_name();
        argument.is_enum = value.has_enum_name();
        argument.first_field = static_cast<uint32_t>(site.payload_codes.size());
        argument.is_array = value.has_array_length();
        // Where its fields lie within an element isn't sent; the decoder needs only how many an element has.
        argument.field_bits.resize(fields / elements);
        for (const proto::PayloadField& field : value.fields()) {
            site.payload_codes.push_back(field_code(field.type()));
            layout.payload_bits.push_back({field.bit_offset(), field.bit_size()});
            payload_end_bits = std::max(payload_end_bits, field.bit_offset() + field.bit_size());
        }
        site.argument_field_counts.push_back(fields);
    }
    layout.packet_words = 3 + (payload_end_bits + 31) / 32;
    site.packet_layouts.push_back(std::move(layout));
    site.sends_to_external_process = true;
    return site;
}

// Everything the decoder reads of the call sites with ids `ids`, whose records go to the external process, which adds
// them to its own.
void describe_call_sites(std::span<const uint32_t> ids, google::protobuf::RepeatedPtrField<proto::CallSite>& out) {
    for (const uint32_t id : ids) {
        describe_call_site(id, *tt::llrt::process_call_sites().find(id), *out.Add());
    }
}

// The external process's socket, and the lock it holds while it listens there, in a directory every user can write: any
// user's process may be the external process, and any user's Metal process connects.
constexpr std::string_view kSocketDirectory = "/tmp/tt_metal_device_observability";
constexpr std::string_view kSocketPath = "/tmp/tt_metal_device_observability/external_process.sock";
constexpr std::string_view kLockPath = "/tmp/tt_metal_device_observability/external_process.lock";

// Creates the socket's directory unless it exists; false when it can't. mkdir applies the umask, and a container's
// mount makes the directory as root, so the mode that opens it to every user is set again. Only the owner or root can;
// another user's process finds it set already.
bool make_socket_directory() {
    if (::mkdir(kSocketDirectory.data(), 0777) != 0 && errno != EEXIST) {
        return false;
    }
    ::chmod(kSocketDirectory.data(), 0777);
    return true;
}

sockaddr_un socket_address() {
    sockaddr_un address{};
    address.sun_family = AF_UNIX;
    static_assert(kSocketPath.size() < sizeof(address.sun_path));
    std::memcpy(address.sun_path, kSocketPath.data(), kSocketPath.size());
    return address;
}

// The connected external process's socket, or -1 when none listens.
int connect_to_external_process() {
    const sockaddr_un address = socket_address();
    const int fd = ::socket(AF_UNIX, SOCK_STREAM | SOCK_CLOEXEC, 0);
    if (fd < 0) {
        return -1;
    }
    if (::connect(fd, reinterpret_cast<const sockaddr*>(&address), sizeof(address)) != 0) {
        ::close(fd);
        return -1;
    }
    // Room for what an external process that isn't running for a while leaves unread, without this process waiting: the
    // largest message, a 32-chip system's descriptor, is about 150 KB, and an image's call sites a few KB. A host caps
    // it at twice net.core.wmem_max, 416 KB by default.
    const int send_buffer_bytes = 1 << 20;
    ::setsockopt(fd, SOL_SOCKET, SO_SNDBUF, &send_buffer_bytes, sizeof(send_buffer_bytes));
    // An external process that stops reading must not hold up this process, whose image loads push call sites: a
    // send that waits this long ends the connection instead.
    const timeval send_timeout{.tv_sec = 1, .tv_usec = 0};
    ::setsockopt(fd, SOL_SOCKET, SO_SNDTIMEO, &send_timeout, sizeof(send_timeout));
    return fd;
}

}  // namespace

proto::FrameLayout this_build_frame_layout() {
    namespace kp = kernel_profiler;
    const auto set = [](proto::BitField& field, uint32_t bit_offset, uint32_t bit_size) {
        field.set_bit_offset(bit_offset);
        field.set_bit_size(bit_size);
    };
    constexpr uint32_t kWordBits = 32;
    constexpr uint32_t kPageFieldBits = std::popcount(kp::SPSC_SPAN_ROUTED_PAGE_MASK);
    proto::FrameLayout layout;
    layout.set_socket_page_size_words(kp::SPSC_SPAN_PAGE_WORDS);
    layout.set_prefix_length_words(kp::SPSC_SPAN_PREFIX_WORDS);
    set(*layout.mutable_frame_payload_length_field(), kp::SPSC_PREFIX_PAYLOAD_WORDS * kWordBits, kWordBits);
    set(*layout.mutable_core_coord_x_field(), kp::SPSC_PREFIX_XY * kWordBits, 16);
    set(*layout.mutable_core_coord_y_field(), kp::SPSC_PREFIX_XY * kWordBits + 16, 16);
    layout.set_control_block_length_words(kp::SPSC_SPAN_WIRE_CTRL_WORDS);
    layout.set_risc_ring_buffer_size_words(profiler::kSpscRingCap);
    layout.set_whole_ring_buffer_threshold_words(kp::SPSC_SPAN_WRAP_IMAGE_MAX_PAD_WORDS);
    set(*layout.mutable_routed_first_page_field(), kp::SPSC_SPAN_ROUTED_FIRST_SHIFT, kPageFieldBits);
    set(*layout.mutable_routed_end_page_field(), kp::SPSC_SPAN_ROUTED_END_SHIFT, kPageFieldBits);
    for (uint32_t risc = 0; risc < kp::PROFILER_SPSC_TENSIX_RISC; risc++) {
        set(*layout.add_packet_range_start_fields(), (kp::SPSC_PREFIX_HEAD_0 + risc) * kWordBits, kWordBits);
        set(*layout.add_packet_range_end_fields(),
            (kp::SPSC_SPAN_PREFIX_WORDS + kp::SPSC_WIRE_TAIL_0 + risc) * kWordBits,
            kWordBits);
    }
    return layout;
}

proto::PacketFormat this_build_packet_format() {
    const auto set = [](proto::BitField& field, uint32_t bit_offset, uint32_t bit_size) {
        field.set_bit_offset(bit_offset);
        field.set_bit_size(bit_size);
    };
    proto::PacketFormat format;
    using kernel_profiler::PacketType;
    using kernel_profiler::ppfmt;
    set(*format.mutable_packet_type_field(), ppfmt::TYPE_SHIFT, 32 - ppfmt::TYPE_SHIFT);
    format.set_record_packet_type(uint32_t(PacketType::Record));
    set(*format.mutable_record_length_field(), ppfmt::PACKET_WORDS_SHIFT, std::popcount(ppfmt::PACKET_WORDS_MASK));
    set(*format.mutable_routed_field(), std::countr_zero(ppfmt::TO_EXTERNAL), 1);
    set(*format.mutable_call_site_id_field(), 0, std::popcount(ppfmt::CALL_SITE_ID_MASK));
    (*format.mutable_non_record_packet_lengths_words())[uint32_t(PacketType::RuntimeId)] = 1;
    (*format.mutable_non_record_packet_lengths_words())[uint32_t(PacketType::TimerHigh)] = 1;
    (*format.mutable_non_record_packet_lengths_words())[uint32_t(PacketType::RuntimeIdWide)] = 2;
    return format;
}

ExternalProcessClient::ExternalProcessClient(const proto::CaptureDescriptor& descriptor, Detach detach) :
    descriptor_(std::make_unique<proto::CaptureDescriptor>(descriptor)), detach_(std::move(detach)) {}

void ExternalProcessClient::end_capture() {
    const std::lock_guard sending(send_mu_);
    capture_ended_ = true;
    if (fd_ >= 0) {
        proto::Envelope envelope;
        envelope.mutable_capture_ended();
        if (!send_message(fd_, serialize(envelope))) {
            ::shutdown(fd_, SHUT_RDWR);
        }
    }
}

void ExternalProcessClient::start() {
    // An image's call sites go out while it loads, so the external process holds them before any of its records,
    // even if this process dies right after.
    tt::llrt::set_call_site_listener([this] {
        const std::lock_guard sending(send_mu_);
        if (fd_ >= 0 && !push_call_sites()) {
            // The external process lacks this image's call sites, so none of the image's records may reach its rings:
            // the relays' writes to them go off before the image can run. The connection's thread then sees it end.
            ::shutdown(fd_, SHUT_RDWR);
            detach_();
        }
    });
    wake_fd_ = ::eventfd(0, EFD_CLOEXEC);
    TT_FATAL(wake_fd_ >= 0, "streaming profiler: eventfd() failed: {}", std::strerror(errno));
    thread_ = std::thread([this] { run(); });
}

ExternalProcessClient::~ExternalProcessClient() {
    tt::llrt::set_call_site_listener({});
    {
        const std::lock_guard lock(mu_);
        stop_ = true;
    }
    if (wake_fd_ >= 0) {
        const uint64_t one = 1;
        [[maybe_unused]] const ssize_t written = ::write(wake_fd_, &one, sizeof(one));
    }
    {
        const std::lock_guard sending(send_mu_);
        if (fd_ >= 0) {
            ::shutdown(fd_, SHUT_RDWR);
        }
    }
    if (thread_.joinable()) {
        thread_.join();
    }
    if (wake_fd_ >= 0) {
        ::close(wake_fd_);
    }
}

// Connects now if an external process listens, and otherwise once one starts listening: its socket appears in the
// socket's directory, which an inotify watch on the directory reports. After a connection ends, the next external
// process is waited for the same way. A watch that goes with its directory is set up again.
void ExternalProcessClient::run() {
    const int watch = ::inotify_init1(IN_NONBLOCK | IN_CLOEXEC);
    const auto watch_directory = [&] {
        if (watch >= 0 && make_socket_directory() &&
            ::inotify_add_watch(watch, kSocketDirectory.data(), IN_CREATE) >= 0) {
            return true;
        }
        log_error(
            tt::LogMetal,
            "[streaming profiler] can't watch {} for an external process, so only one already listening gets this "
            "capture's records: {}",
            kSocketDirectory,
            std::strerror(errno));
        return false;
    };
    bool watching = watch_directory();
    while (true) {
        if (const std::lock_guard lock(mu_); stop_) {
            break;
        }
        if (const int fd = connect_to_external_process(); fd >= 0) {
            serve(fd);
        }
        if (!watching) {
            break;
        }
        std::array<pollfd, 2> fds{{{watch, POLLIN, 0}, {wake_fd_, POLLIN, 0}}};
        while (::poll(fds.data(), fds.size(), -1) < 0 && errno == EINTR) {
        }
        if (fds[1].revents != 0) {
            break;
        }
        alignas(inotify_event) char events[4096];
        for (ssize_t bytes = 0; (bytes = ::read(watch, events, sizeof(events))) > 0;) {
            for (ssize_t at = 0; at < bytes;) {
                const auto* event = reinterpret_cast<const inotify_event*>(events + at);
                if ((event->mask & IN_IGNORED) != 0) {
                    watching = watch_directory();
                }
                at += static_cast<ssize_t>(sizeof(inotify_event) + event->len);
            }
        }
    }
    if (watch >= 0) {
        ::close(watch);
    }
}

bool ExternalProcessClient::push_call_sites() {
    const std::vector<uint32_t> ids = tt::llrt::ids_of_call_sites_sending_to_external_process(call_sites_pushed_);
    if (ids.empty()) {
        return true;
    }
    proto::Envelope envelope;
    describe_call_sites(ids, *envelope.mutable_call_sites()->mutable_call_sites());
    if (!send_message(fd_, serialize(envelope))) {
        return false;
    }
    call_sites_pushed_ += ids.size();
    return true;
}

// Sends the capture descriptor and the call sites loaded so far, then holds the connection until either side closes it.
// A capture that has already ended isn't served.
void ExternalProcessClient::serve(int fd) {
    {
        const std::lock_guard sending(send_mu_);
        // The destructor shuts down the connection it finds here, so a connection it can't find isn't served.
        if (const std::lock_guard lock(mu_); stop_) {
            ::close(fd);
            return;
        }
        if (capture_ended_) {
            ::close(fd);
            return;
        }
        fd_ = fd;
        // The descriptor carries every call site loaded so far, which the external process then holds before it points
        // the devices at its rings. An image that loads later sends its own before it can run (push_call_sites).
        proto::Envelope envelope;
        proto::CaptureDescriptor& descriptor = *envelope.mutable_capture_descriptor();
        descriptor = *descriptor_;
        const std::vector<uint32_t> ids = tt::llrt::ids_of_call_sites_sending_to_external_process(0);
        describe_call_sites(ids, *descriptor.mutable_call_sites());
        call_sites_pushed_ = ids.size();
        if (!send_message(fd, serialize(envelope))) {
            ::shutdown(fd, SHUT_RDWR);
        }
    }
    // The external process sends nothing, so this returns once it closes the connection or this process shuts it down.
    char byte = 0;
    while (::recv(fd, &byte, 1, 0) < 0 && errno == EINTR) {
    }
    {
        const std::lock_guard sending(send_mu_);
        fd_ = -1;
    }
    // The external process may have gone without turning the relays' writes off.
    detach_();
    ::close(fd);
}

ExternalListener::ExternalListener() : fd_(::socket(AF_UNIX, SOCK_STREAM | SOCK_CLOEXEC, 0)) {}

ExternalListener::~ExternalListener() {
    ::close(fd_);
    ::close(lock_fd_);
}

bool ExternalListener::listen() {
    TT_FATAL(
        make_socket_directory(), "streaming profiler: can't create {}: {}", kSocketDirectory, std::strerror(errno));
    // open applies the umask, so the mode that opens the lock to every user is set again, as the directory's is.
    lock_fd_ = ::open(kLockPath.data(), O_RDWR | O_CREAT | O_CLOEXEC, 0666);
    TT_FATAL(lock_fd_ >= 0, "streaming profiler: can't open {}: {}", kLockPath, std::strerror(errno));
    ::fchmod(lock_fd_, 0666);  // fails, harmlessly, on another user's lock, whose process already did it
    if (::flock(lock_fd_, LOCK_EX | LOCK_NB) != 0) {
        TT_FATAL(errno == EWOULDBLOCK, "streaming profiler: can't lock {}: {}", kLockPath, std::strerror(errno));
        ::close(std::exchange(lock_fd_, -1));
        return false;
    }
    // The lock is this process's now, so a socket file here is one an external process that died left.
    ::unlink(kSocketPath.data());
    const sockaddr_un address = socket_address();
    TT_FATAL(
        ::bind(fd_, reinterpret_cast<const sockaddr*>(&address), sizeof(address)) == 0 &&
            ::chmod(kSocketPath.data(), 0666) == 0 && ::listen(fd_, 4) == 0,
        "streaming profiler: can't listen on {}: {}",
        kSocketPath,
        std::strerror(errno));
    return true;
}

int ExternalListener::accept() const {
    while (true) {
        const int fd = ::accept4(fd_, nullptr, nullptr, SOCK_CLOEXEC);
        if (fd < 0 && (errno == EINTR || errno == ECONNABORTED)) {
            continue;
        }
        return fd;
    }
}

ClientConnection::ClientConnection(int fd) : fd_(fd), call_sites_(std::make_shared<tt::llrt::CallSites>()) {
    ucred peer{};
    socklen_t size = sizeof(peer);
    if (::getsockopt(fd_, SOL_SOCKET, SO_PEERCRED, &peer, &size) == 0) {
        pid_ = peer.pid;
    }
    // A Metal process sends its capture descriptor as soon as it connects; one that doesn't in time is dropped.
    ::setsockopt(fd_, SOL_SOCKET, SO_RCVTIMEO, &kDescriptorTimeout, sizeof(kDescriptorTimeout));
    try {
        proto::Envelope envelope;
        if (const auto first = recv_message(fd_);
            first && envelope.ParseFromArray(first->data(), static_cast<int>(first->size())) &&
            envelope.has_capture_descriptor()) {
            if (envelope.capture_descriptor().protocol_version() == kProtocolVersion) {
                descriptor_ =
                    std::make_unique<proto::CaptureDescriptor>(std::move(*envelope.mutable_capture_descriptor()));
                // The call sites loaded before the connection, here before any device writes to this process's rings.
                for (const proto::CallSite& call_site : descriptor_->call_sites()) {
                    call_sites_->add(call_site.id(), read_call_site(call_site));
                }
            } else {
                log_error(
                    tt::LogMetal,
                    "[streaming profiler] Metal process {} speaks protocol version {}, and this process {}",
                    pid_,
                    envelope.capture_descriptor().protocol_version(),
                    kProtocolVersion);
            }
        }
        if (!descriptor_) {
            closed_.store(true, std::memory_order_release);
        }
    } catch (...) {
        ::close(fd_);
        throw;
    }
    const timeval no_timeout{};
    ::setsockopt(fd_, SOL_SOCKET, SO_RCVTIMEO, &no_timeout, sizeof(no_timeout));
}

ClientConnection::~ClientConnection() {
    ::shutdown(fd_, SHUT_RDWR);
    if (thread_.joinable()) {
        thread_.join();
    }
    ::close(fd_);
}

void ClientConnection::start() {
    thread_ = std::thread([this] { read_thread(); });
}

// A connection that fails here ends, and this process goes on to serve the next Metal process.
void ClientConnection::read_thread() {
    try {
        while (const auto received = recv_message(fd_)) {
            proto::Envelope envelope;
            TT_FATAL(
                envelope.ParseFromArray(received->data(), static_cast<int>(received->size())),
                "external connection: a message didn't parse");
            if (envelope.has_capture_ended()) {
                capture_ended_.store(true, std::memory_order_release);
                continue;
            }
            TT_FATAL(envelope.has_call_sites(), "external connection: an unexpected message");
            {
                const std::lock_guard lock(sites_mu_);
                for (const proto::CallSite& call_site : envelope.call_sites().call_sites()) {
                    call_sites_->add(call_site.id(), read_call_site(call_site));
                }
            }
            sites_cv_.notify_all();
        }
    } catch (const std::exception& e) {
        log_error(tt::LogMetal, "[streaming profiler] dropping the connection to Metal process {}: {}", pid_, e.what());
        drop();
    }
    std::lock_guard lock(sites_mu_);
    closed_.store(true, std::memory_order_release);
    sites_cv_.notify_all();
}

void ClientConnection::drop() {
    dropped_.store(true, std::memory_order_release);
    ::shutdown(fd_, SHUT_RDWR);
}

const tt::llrt::CallSite& ClientConnection::wait_for_call_site(uint32_t id) {
    const tt::llrt::CallSite* site = call_sites_->find(id);
    if (site == nullptr) {
        std::unique_lock lock(sites_mu_);
        sites_cv_.wait(lock, [&] { return (site = call_sites_->find(id)) != nullptr || closed(); });
    }
    TT_FATAL(
        site != nullptr,
        "streaming profiler: Metal process {} sent a record of call site {}, which it never described",
        pid_,
        id);
    return *site;
}

}  // namespace tt::tt_metal::streaming_profiler::external
