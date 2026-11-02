// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "impl/streaming_profiler/external_connection.hpp"

#include <fcntl.h>
#include <poll.h>
#include <sys/file.h>
#include <sys/socket.h>
#include <sys/stat.h>
#include <sys/un.h>
#include <unistd.h>

#include <array>
#include <cerrno>
#include <chrono>
#include <cstddef>
#include <cstring>
#include <memory>
#include <span>
#include <string>
#include <utility>
#include <variant>

#include <tt_stl/assert.hpp>

#include "llrt/call_sites.hpp"

namespace tt::tt_metal::streaming_profiler::external {

namespace {

namespace api = tt::tt_metal::experimental::device_observability;

// Messages are a 32-bit byte count and the bytes, which start with the Message. A request gets one kReply back.
enum Message : uint32_t { kHello = 0, kClock = 1, kReply = 2, kSite = 3, kAttach = 4, kDetach = 5 };

// How long a Metal process opening its devices waits for an external process that listens to attach.
constexpr auto kFirstAttachWait = std::chrono::seconds(10);
// How often a Metal process with no external process connected tries again.
constexpr int kReconnectMs = 1000;

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
    if (!recv_all(fd, std::as_writable_bytes(std::span(&bytes, 1)))) {
        return std::nullopt;
    }
    std::vector<uint8_t> message(bytes);
    if (!recv_all(fd, std::as_writable_bytes(std::span(message)))) {
        return std::nullopt;
    }
    return message;
}

struct Writer {
    std::vector<uint8_t> out;
    template <typename T>
    void put(T value) {
        const auto* p = reinterpret_cast<const uint8_t*>(&value);
        out.insert(out.end(), p, p + sizeof(value));
    }
    void u32(uint32_t value) { put(value); }
    void bytes(std::span<const uint8_t> data) {
        u32(uint32_t(data.size()));
        out.insert(out.end(), data.begin(), data.end());
    }
    void text(std::string_view s) { bytes({reinterpret_cast<const uint8_t*>(s.data()), s.size()}); }
    template <typename T>
    void list(std::span<const T> items) {
        u32(uint32_t(items.size()));
        for (const T& item : items) {
            put(item);
        }
    }
};
struct Reader {
    std::span<const uint8_t> in;
    template <typename T>
    T get() {
        TT_FATAL(in.size() >= sizeof(T), "external connection: truncated message");
        T value;
        std::memcpy(&value, in.data(), sizeof(value));
        in = in.subspan(sizeof(value));
        return value;
    }
    uint32_t u32() { return get<uint32_t>(); }
    std::span<const uint8_t> bytes() {
        const uint32_t n = u32();
        TT_FATAL(in.size() >= n, "external connection: truncated message");
        const auto data = in.first(n);
        in = in.subspan(n);
        return data;
    }
    std::string_view text() {
        const auto data = bytes();
        return {reinterpret_cast<const char*>(data.data()), data.size()};
    }
    template <typename T>
    std::vector<T> list() {
        std::vector<T> items(u32());
        for (T& item : items) {
            item = get<T>();
        }
        return items;
    }
};

Writer message(Message kind) {
    Writer w;
    w.u32(kind);
    return w;
}

void put_device(Writer& w, const CaptureContext::Device& device) {
    w.u32(device.chip_id);
    w.u32(device.start_timer_hi);
    w.put(device.check_offset);
    w.u32(uint32_t(device.lanes.size()));
    for (const api::Core& core : device.lanes) {
        w.u32(uint32_t(core.logical.x));
        w.u32(uint32_t(core.logical.y));
        w.u32(uint32_t(core.physical.x));
        w.u32(uint32_t(core.physical.y));
        w.u32(uint32_t(core.chip_id));
        w.u32(uint32_t(core.processor));
    }
    w.list(std::span<const int64_t>(device.clock_offsets));
    std::vector<uint32_t> cores;  // (slot, core) of each seeded slot
    for (uint32_t slot = 0; slot < device.core_of_xy.slot.size(); slot++) {
        if (device.core_of_xy.slot[slot] != CoreTable::kNone) {
            cores.push_back(slot);
            cores.push_back(device.core_of_xy.slot[slot]);
        }
    }
    w.list(std::span<const uint32_t>(cores));
}
CaptureContext::Device get_device(Reader& r) {
    CaptureContext::Device device;
    device.chip_id = r.u32();
    device.start_timer_hi = r.u32();
    device.check_offset = r.get<int64_t>();
    device.lanes.resize(r.u32());
    for (api::Core& core : device.lanes) {
        core.logical.x = r.u32();
        core.logical.y = r.u32();
        core.physical.x = r.u32();
        core.physical.y = r.u32();
        core.chip_id = ChipId(r.u32());
        core.processor = api::Processor(r.u32());
    }
    device.clock_offsets = r.list<int64_t>();
    const std::vector<uint32_t> cores = r.list<uint32_t>();
    for (size_t i = 0; i + 1 < cores.size(); i += 2) {
        device.core_of_xy.slot.at(cores[i]) = uint16_t(cores[i + 1]);
    }
    return device;
}

std::vector<uint8_t> encode_hello(const Hello& hello) {
    Writer w = message(kHello);
    w.u32(uint32_t(hello.relays.size()));
    for (const Hello::Relay& relay : hello.relays) {
        w.u32(relay.pcie_device);
        w.u32(relay.dev);
    }
    w.put(hello.bases.root_refclk);
    w.put(hello.bases.tsc);
    w.u32(uint32_t(hello.devices.size()));
    for (const CaptureContext::Device& device : hello.devices) {
        put_device(w, device);
    }
    return std::move(w.out);
}
Hello decode_hello(Reader& r) {
    Hello hello;
    hello.relays.resize(r.u32());
    for (Hello::Relay& relay : hello.relays) {
        relay.pcie_device = r.u32();
        relay.dev = r.u32();
    }
    hello.bases.root_refclk = r.get<int64_t>();
    hello.bases.tsc = r.get<int64_t>();
    hello.devices.resize(r.u32());
    for (CaptureContext::Device& device : hello.devices) {
        device = get_device(r);
    }
    return hello;
}

void put_update(Writer& w, const ClockUpdate& update) {
    w.u32(uint32_t(update.kind));
    w.u32(update.dev);
    w.put(update.chip);
    w.put(update.host);
    w.put(update.until);
}
ClockUpdate get_update(Reader& r) {
    ClockUpdate update;
    update.kind = ClockUpdate::Kind(r.u32());
    update.dev = r.u32();
    update.chip = r.get<SyncNode>();
    update.host = r.get<HostNode>();
    update.until = r.get<double>();
    return update;
}

void put_bits(Writer& w, std::span<const tt::llrt::FieldBits> bits) {
    w.u32(uint32_t(bits.size()));
    for (const tt::llrt::FieldBits& b : bits) {
        w.u32(b.bit);
        w.u32(b.width);
    }
}
std::vector<tt::llrt::FieldBits> get_bits(Reader& r) {
    std::vector<tt::llrt::FieldBits> bits(r.u32());
    for (tt::llrt::FieldBits& b : bits) {
        b.bit = r.u32();
        b.width = r.u32();
    }
    return bits;
}

// Everything the decoder reads of a call site, which the external process adds to its own.
std::vector<uint8_t> encode_site(uint32_t id) {
    Writer w = message(kReply);
    const tt::llrt::CallSite* site = id <= 0xFFFFu ? tt::llrt::process_call_sites().find(id) : nullptr;
    w.u32(site != nullptr);
    if (site == nullptr) {
        return std::move(w.out);
    }
    w.text(site->location.file);
    w.text(site->location.function);
    w.u32(site->location.line);
    w.text(site->metadata_type);
    w.list(std::span<const uint16_t>(site->payload_codes));
    w.list(std::span<const uint16_t>(site->metadata_codes));
    put_bits(w, site->metadata_bits);
    w.u32(uint32_t(site->arguments.size()));
    for (const tt::llrt::CallSite::Argument& argument : site->arguments) {
        w.text(argument.type);
        w.u32(argument.first_field);
        w.u32(argument.is_array);
        put_bits(w, argument.field_bits);
    }
    w.list(std::span<const uint32_t>(site->argument_field_counts));
    w.u32(uint32_t(site->packet_layouts.size()));
    for (const tt::llrt::CallSite::PacketLayout& layout : site->packet_layouts) {
        w.u32(uint32_t(layout.time_form));
        w.u32(layout.body_words);
        put_bits(w, layout.payload_bits);
    }
    w.u32(site->timestamps_as_distances);
    w.u32(site->payload_publish_time_field);
    w.u32(uint32_t(site->metadata.size()));
    for (const auto& value : site->metadata) {
        const auto* text = std::get_if<std::string_view>(&value);
        w.u32(text != nullptr);
        if (text != nullptr) {
            w.text(*text);
        } else {
            w.put(std::get<uint64_t>(value));
        }
    }
    return std::move(w.out);
}
// The site's names point into `r`'s message, which CallSites::add stores.
std::optional<tt::llrt::CallSite> decode_site(Reader& r) {
    if (r.u32() == 0) {
        return std::nullopt;
    }
    tt::llrt::CallSite site;
    site.location.file = r.text();
    site.location.function = r.text();
    site.location.line = r.u32();
    site.metadata_type = r.text();
    site.payload_codes = r.list<uint16_t>();
    site.metadata_codes = r.list<uint16_t>();
    site.metadata_bits = get_bits(r);
    site.arguments.resize(r.u32());
    for (tt::llrt::CallSite::Argument& argument : site.arguments) {
        argument.type = r.text();
        argument.first_field = r.u32();
        argument.is_array = r.u32() != 0;
        argument.field_bits = get_bits(r);
    }
    site.argument_field_counts = r.list<uint32_t>();
    site.packet_layouts.resize(r.u32());
    for (tt::llrt::CallSite::PacketLayout& layout : site.packet_layouts) {
        layout.time_form = api::detail::TimeForm(r.u32());
        layout.body_words = r.u32();
        layout.payload_bits = get_bits(r);
    }
    site.timestamps_as_distances = r.u32() != 0;
    site.payload_publish_time_field = r.u32();
    site.metadata.resize(r.u32());
    for (auto& value : site.metadata) {
        if (r.u32() != 0) {
            value = r.text();
        } else {
            value = r.get<uint64_t>();
        }
    }
    return site;
}

// The external process's socket, and the lock it holds while it listens there, in a directory every user can write: any
// user's process may be the external process, and any user's Metal process connects.
constexpr std::string_view kSocketDirectory = "/tmp/tt_metal_device_observability";
constexpr std::string_view kSocketPath = "/tmp/tt_metal_device_observability/external_process.sock";
constexpr std::string_view kLockPath = "/tmp/tt_metal_device_observability/external_process.lock";

sockaddr_un socket_address() {
    sockaddr_un address{};
    address.sun_family = AF_UNIX;
    static_assert(kSocketPath.size() < sizeof(address.sun_path));
    std::memcpy(address.sun_path, kSocketPath.data(), kSocketPath.size());
    return address;
}

int connect_to_external_process() {
    const sockaddr_un address = socket_address();
    const int fd = ::socket(AF_UNIX, SOCK_STREAM | SOCK_CLOEXEC, 0);
    TT_FATAL(fd >= 0, "external connection: socket() failed: {}", std::strerror(errno));
    if (::connect(fd, reinterpret_cast<const sockaddr*>(&address), sizeof(address)) != 0) {
        ::close(fd);
        return -1;
    }
    return fd;
}

}  // namespace

ExternalProcessClient::ExternalProcessClient(const Hello& hello, Attach attach, Detach detach) :
    hello_(encode_hello(hello)),
    attach_(std::move(attach)),
    detach_(std::move(detach)),
    newest_chip_(hello.devices.size()),
    finished_(hello.devices.size(), false) {
    TT_FATAL(
        ::pipe2(wake_, O_CLOEXEC | O_NONBLOCK) == 0, "external connection: pipe2() failed: {}", std::strerror(errno));
}

void ExternalProcessClient::start() {
    thread_ = std::thread([this] { run(); });
    std::unique_lock lock(mu_);
    started_cv_.wait_for(lock, kFirstAttachWait, [&] { return start_ == Start::Done; });
}

ExternalProcessClient::~ExternalProcessClient() {
    {
        std::lock_guard lock(mu_);
        stop_ = true;
    }
    wake();
    if (thread_.joinable()) {
        thread_.join();
    }
    ::close(wake_[0]);
    ::close(wake_[1]);
}

void ExternalProcessClient::wake() {
    const char byte = 0;
    [[maybe_unused]] const ssize_t written = ::write(wake_[1], &byte, 1);
}

void ExternalProcessClient::send(const ClockUpdate& update) {
    bool first = false;
    {
        std::lock_guard lock(mu_);
        switch (update.kind) {
            case ClockUpdate::Kind::Chip: newest_chip_[update.dev] = update; break;
            case ClockUpdate::Kind::Host: newest_host_ = update; break;
            case ClockUpdate::Kind::Finish: finished_[update.dev] = true; break;
        }
        if (!connected_) {
            return;
        }
        first = queued_.empty();
        queued_.push_back(update);
    }
    if (first) {
        wake();
    }
}

void ExternalProcessClient::run() {
    while (true) {
        if (std::lock_guard lock(mu_); stop_) {
            return;
        }
        if (const int fd = connect_to_external_process(); fd >= 0) {
            serve(fd);
        }
        // An external process that wasn't listening when the devices opened, or has gone, leaves the open waiting no
        // more.
        {
            std::lock_guard lock(mu_);
            start_ = Start::Done;
            started_cv_.notify_all();
        }
        pollfd waker{wake_[0], POLLIN, 0};
        if (::poll(&waker, 1, kReconnectMs) > 0) {
            char drain[64];
            while (::read(wake_[0], drain, sizeof(drain)) > 0) {
            }
        }
    }
}

// Sends the hello and the clock map's newest state, then answers the external process's requests and forwards the
// clock map's changes until either side closes the connection.
void ExternalProcessClient::serve(int fd) {
    std::vector<uint32_t> attached;
    bool open = send_message(fd, hello_);
    {
        Writer snapshot = message(kClock);
        std::vector<ClockUpdate> updates;
        {
            std::lock_guard lock(mu_);
            connected_ = true;
            queued_.clear();
            if (newest_host_) {
                updates.push_back(*newest_host_);
            }
            for (uint32_t dev = 0; dev < newest_chip_.size(); dev++) {
                if (newest_chip_[dev]) {
                    updates.push_back(*newest_chip_[dev]);
                }
                if (finished_[dev]) {
                    updates.push_back({.kind = ClockUpdate::Kind::Finish, .dev = dev});
                }
            }
        }
        snapshot.u32(uint32_t(updates.size()));
        for (const ClockUpdate& update : updates) {
            put_update(snapshot, update);
        }
        open = open && send_message(fd, snapshot.out);
    }
    std::array<pollfd, 2> fds{{{fd, POLLIN, 0}, {wake_[0], POLLIN, 0}}};
    while (open) {
        if (::poll(fds.data(), fds.size(), -1) < 0) {
            open = errno == EINTR;
            continue;
        }
        if (fds[1].revents != 0) {
            char drain[64];
            while (::read(wake_[0], drain, sizeof(drain)) > 0) {
            }
            std::vector<ClockUpdate> updates;
            bool stopping = false;
            {
                std::lock_guard lock(mu_);
                updates.swap(queued_);
                stopping = stop_;
            }
            if (!updates.empty()) {
                Writer w = message(kClock);
                w.u32(uint32_t(updates.size()));
                for (const ClockUpdate& update : updates) {
                    put_update(w, update);
                }
                open = send_message(fd, w.out);
            }
            if (stopping) {
                break;
            }
        }
        if (fds[0].revents == 0) {
            continue;
        }
        const auto request = (fds[0].revents & POLLIN) != 0 ? recv_message(fd) : std::nullopt;
        if (!request.has_value()) {
            break;
        }
        Reader r{*request};
        std::vector<uint8_t> reply;
        switch (r.u32()) {
            case kSite: reply = encode_site(r.u32()); break;
            case kAttach: {
                bool ok = true;
                for (uint32_t n = r.u32(); n != 0; n--) {
                    RelayBuffer buffer;
                    buffer.relay = r.u32();
                    buffer.noc_addr = r.get<uint64_t>();
                    buffer.bytes = r.u32();
                    if (attach_(buffer)) {
                        attached.push_back(buffer.relay);
                    } else {
                        ok = false;
                    }
                }
                Writer w = message(kReply);
                w.u32(ok);
                reply = std::move(w.out);
                std::lock_guard lock(mu_);
                start_ = Start::Done;
                started_cv_.notify_all();
                break;
            }
            case kDetach: {
                for (const uint32_t relay : attached) {
                    detach_(relay);
                }
                attached.clear();
                reply = message(kReply).out;
                break;
            }
            default: TT_THROW("external connection: unknown request");
        }
        open = send_message(fd, reply);
    }
    for (const uint32_t relay : attached) {
        detach_(relay);
    }
    {
        std::lock_guard lock(mu_);
        connected_ = false;
        queued_.clear();
    }
    ::close(fd);
}

ExternalListener::ExternalListener() : fd_(::socket(AF_UNIX, SOCK_STREAM | SOCK_CLOEXEC, 0)) {}

ExternalListener::~ExternalListener() {
    ::close(fd_);
    ::close(lock_fd_);
}

bool ExternalListener::listen() {
    // mkdir and open apply the umask, and a container's mount makes the directory as root, so the modes that open them
    // to every user are set again. Only the owner or root can; another user's process finds them set already.
    TT_FATAL(
        ::mkdir(kSocketDirectory.data(), 0777) == 0 || errno == EEXIST,
        "streaming profiler: can't create {}: {}",
        kSocketDirectory,
        std::strerror(errno));
    ::chmod(kSocketDirectory.data(), 0777);
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
        if (fd >= 0 || (errno != EINTR && errno != ECONNABORTED)) {
            return fd;
        }
    }
}

void ExternalListener::stop() { ::shutdown(fd_, SHUT_RDWR); }

ClientConnection::ClientConnection(int fd) : fd_(fd), call_sites_(std::make_shared<tt::llrt::CallSites>()) {
    if (const auto first = recv_message(fd_)) {
        Reader r{*first};
        TT_FATAL(r.u32() == kHello, "external connection: the Metal process sent no hello");
        hello_ = decode_hello(r);
    } else {
        closed_.store(true, std::memory_order_release);
    }
}

ClientConnection::~ClientConnection() {
    ::shutdown(fd_, SHUT_RDWR);
    if (thread_.joinable()) {
        thread_.join();
    }
    ::close(fd_);
}

void ClientConnection::start(OnClock on_clock) {
    on_clock_ = std::move(on_clock);
    thread_ = std::thread([this] { read_thread(); });
}

void ClientConnection::read_thread() {
    std::vector<ClockUpdate> updates;
    while (const auto received = recv_message(fd_)) {
        Reader r{*received};
        switch (r.u32()) {
            case kClock: {
                updates.resize(r.u32());
                for (ClockUpdate& update : updates) {
                    update = get_update(r);
                }
                on_clock_(updates);
                break;
            }
            case kReply: {
                std::lock_guard lock(reply_mu_);
                reply_ = std::vector<uint8_t>(r.in.begin(), r.in.end());
                reply_cv_.notify_all();
                break;
            }
            default: TT_THROW("external connection: unknown message");
        }
    }
    std::lock_guard lock(reply_mu_);
    closed_.store(true, std::memory_order_release);
    reply_cv_.notify_all();
}

std::optional<std::vector<uint8_t>> ClientConnection::request(const std::vector<uint8_t>& message) {
    const std::lock_guard one_at_a_time(request_mu_);
    if (closed() || !send_message(fd_, message)) {
        return std::nullopt;
    }
    std::unique_lock lock(reply_mu_);
    reply_cv_.wait(lock, [&] { return reply_.has_value() || closed(); });
    return std::exchange(reply_, std::nullopt);
}

bool ClientConnection::attach(std::span<const RelayBuffer> buffers) {
    Writer w = message(kAttach);
    w.u32(uint32_t(buffers.size()));
    for (const RelayBuffer& buffer : buffers) {
        w.u32(buffer.relay);
        w.put(buffer.noc_addr);
        w.u32(buffer.bytes);
    }
    const auto reply = request(w.out);
    return reply.has_value() && Reader{*reply}.u32() != 0;
}

void ClientConnection::detach() { static_cast<void>(request(message(kDetach).out)); }

const tt::llrt::CallSite* ClientConnection::add_site(uint32_t id) {
    Writer w = message(kSite);
    w.u32(id);
    const auto reply = request(w.out);
    if (!reply) {
        return nullptr;
    }
    Reader r{*reply};
    std::optional<tt::llrt::CallSite> site = decode_site(r);
    if (!site) {
        return nullptr;
    }
    call_sites_->add(id, std::move(*site));
    return call_sites_->find(id);
}

}  // namespace tt::tt_metal::streaming_profiler::external
