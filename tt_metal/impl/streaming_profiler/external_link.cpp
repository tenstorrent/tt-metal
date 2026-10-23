// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "impl/streaming_profiler/external_link.hpp"

#include <fcntl.h>
#include <poll.h>
#include <sys/socket.h>
#include <sys/time.h>
#include <sys/un.h>
#include <unistd.h>

#include <algorithm>
#include <array>
#include <cerrno>
#include <cstddef>
#include <cstring>
#include <memory>
#include <string>
#include <utility>

#include <tt-logger/tt-logger.hpp>
#include <tt_stl/assert.hpp>

#include "llrt/call_sites.hpp"

namespace tt::tt_metal::streaming_profiler::external {

namespace {

namespace sp = tt::tt_metal::experimental::streaming_profiler;

// Messages are a 32-bit byte count and the bytes, which start with the Message.
enum Message : uint32_t { kHello = 0, kSitePush = 1 };

// How often a profiled process with no external process connected tries again.
constexpr int kReconnectMs = 1000;

bool send_all(int fd, const void* data, size_t bytes) {
    const auto* p = static_cast<const char*>(data);
    while (bytes != 0) {
        const ssize_t n = ::send(fd, p, bytes, MSG_NOSIGNAL);
        if (n <= 0) {
            if (n < 0 && errno == EINTR) {
                continue;
            }
            return false;
        }
        p += n;
        bytes -= size_t(n);
    }
    return true;
}
bool recv_all(int fd, void* data, size_t bytes) {
    auto* p = static_cast<char*>(data);
    while (bytes != 0) {
        const ssize_t n = ::recv(fd, p, bytes, 0);
        if (n <= 0) {
            if (n < 0 && errno == EINTR) {
                continue;
            }
            return false;
        }
        p += n;
        bytes -= size_t(n);
    }
    return true;
}
bool send_message(int fd, const std::vector<uint8_t>& message) {
    const auto bytes = uint32_t(message.size());
    return send_all(fd, &bytes, sizeof(bytes)) && send_all(fd, message.data(), message.size());
}
std::optional<std::vector<uint8_t>> recv_message(int fd) {
    uint32_t bytes = 0;
    if (!recv_all(fd, &bytes, sizeof(bytes))) {
        return std::nullopt;
    }
    std::vector<uint8_t> message(bytes);
    if (!recv_all(fd, message.data(), bytes)) {
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
        TT_FATAL(in.size() >= sizeof(T), "external link: truncated message");
        T value;
        std::memcpy(&value, in.data(), sizeof(value));
        in = in.subspan(sizeof(value));
        return value;
    }
    uint32_t u32() { return get<uint32_t>(); }
    std::span<const uint8_t> bytes() {
        const uint32_t n = u32();
        TT_FATAL(in.size() >= n, "external link: truncated message");
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
    for (const sp::Core& core : device.lanes) {
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
    for (sp::Core& core : device.lanes) {
        core.logical.x = r.u32();
        core.logical.y = r.u32();
        core.physical.x = r.u32();
        core.physical.y = r.u32();
        core.chip_id = ChipId(r.u32());
        core.processor = sp::Processor(r.u32());
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
        w.u32(relay.core_x);
        w.u32(relay.core_y);
        w.put(relay.ctrl);
        w.put(relay.config);
        w.u32(relay.pcie_encoding);
    }
    w.put(hello.bases.root_refclk);
    w.put(hello.bases.tsc);
    w.u32(uint32_t(hello.devices.size()));
    for (const CaptureContext::Device& device : hello.devices) {
        put_device(w, device);
    }
    w.u32(uint32_t(hello.links.size()));
    for (const CaptureContext::Link& link : hello.links) {
        for (const uint32_t word : {link.dev_a, link.dev_b, link.core_a, link.core_b}) {
            w.u32(word);
        }
        for (const CoreCoord& eth : {link.eth_a, link.eth_b}) {
            w.u32(uint32_t(eth.x));
            w.u32(uint32_t(eth.y));
        }
    }
    w.u32(hello.root_pcie_x);
    w.u32(hello.root_pcie_y);
    w.u32(hello.l1_alignment);
    return std::move(w.out);
}
Hello decode_hello(Reader& r) {
    Hello hello;
    hello.relays.resize(r.u32());
    for (Hello::Relay& relay : hello.relays) {
        relay.pcie_device = r.u32();
        relay.dev = r.u32();
        relay.core_x = r.u32();
        relay.core_y = r.u32();
        relay.ctrl = r.get<uint64_t>();
        relay.config = r.get<uint64_t>();
        relay.pcie_encoding = r.u32();
    }
    hello.bases.root_refclk = r.get<int64_t>();
    hello.bases.tsc = r.get<int64_t>();
    hello.devices.resize(r.u32());
    for (CaptureContext::Device& device : hello.devices) {
        device = get_device(r);
    }
    hello.links.resize(r.u32());
    for (CaptureContext::Link& link : hello.links) {
        link.dev_a = r.u32();
        link.dev_b = r.u32();
        link.core_a = r.u32();
        link.core_b = r.u32();
        for (CoreCoord* eth : {&link.eth_a, &link.eth_b}) {
            eth->x = r.u32();
            eth->y = r.u32();
        }
    }
    hello.root_pcie_x = r.u32();
    hello.root_pcie_y = r.u32();
    hello.l1_alignment = r.u32();
    return hello;
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
void put_site(Writer& w, const tt::llrt::CallSite& site) {
    w.text(site.location.file);
    w.text(site.location.function);
    w.u32(site.location.line);
    w.text(site.metadata_type);
    w.list(std::span<const uint16_t>(site.data_codes));
    w.list(std::span<const uint16_t>(site.metadata_codes));
    put_bits(w, site.metadata_bits);
    w.u32(uint32_t(site.values.size()));
    for (const tt::llrt::CallSite::Value& value : site.values) {
        w.text(value.type);
        w.u32(value.first_field);
        w.u32(value.is_array);
        put_bits(w, value.field_bits);
    }
    w.list(std::span<const uint32_t>(site.value_fields));
    w.u32(uint32_t(site.forms.size()));
    for (const tt::llrt::CallSite::Form& form : site.forms) {
        w.u32(uint32_t(form.time_form));
        w.u32(form.body_words);
        put_bits(w, form.data_bits);
    }
    w.u32(site.timestamps_as_distances);
    w.u32(site.published_field);
    w.bytes({reinterpret_cast<const uint8_t*>(site.metadata.data()), site.metadata.size()});
    w.u32(site.strings != nullptr);
    if (site.strings != nullptr) {
        w.bytes({reinterpret_cast<const uint8_t*>(site.strings->text.data()), site.strings->text.size()});
        w.put(site.strings->vma);
    }
}

// A call site the external process may meet: one whose records go there, which take only the full form.
bool is_external_site(const tt::llrt::CallSite& site) {
    return !site.forms.empty() && std::ranges::all_of(site.forms, [](const tt::llrt::CallSite::Form& form) {
        return form.time_form == sp::detail::TimeForm::Full;
    });
}
// The site's names point into `r`'s message, which add_call_site copies.
tt::llrt::CallSite decode_site(Reader& r) {
    tt::llrt::CallSite site;
    site.location.file = r.text();
    site.location.function = r.text();
    site.location.line = r.u32();
    site.metadata_type = r.text();
    site.data_codes = r.list<uint16_t>();
    site.metadata_codes = r.list<uint16_t>();
    site.metadata_bits = get_bits(r);
    site.values.resize(r.u32());
    for (tt::llrt::CallSite::Value& value : site.values) {
        value.type = r.text();
        value.first_field = r.u32();
        value.is_array = r.u32() != 0;
        value.field_bits = get_bits(r);
    }
    site.value_fields = r.list<uint32_t>();
    site.forms.resize(r.u32());
    for (tt::llrt::CallSite::Form& form : site.forms) {
        form.time_form = sp::detail::TimeForm(r.u32());
        form.body_words = r.u32();
        form.data_bits = get_bits(r);
    }
    site.timestamps_as_distances = r.u32() != 0;
    site.published_field = r.u32();
    const auto metadata = r.bytes();
    site.metadata.resize(metadata.size());
    std::memcpy(site.metadata.data(), metadata.data(), metadata.size());
    if (r.u32() != 0) {
        auto strings = std::make_shared<tt::llrt::CallSiteStrings>();
        const auto text = r.bytes();
        strings->text.assign(text.begin(), text.end());
        strings->vma = r.get<uint64_t>();
        site.strings = std::move(strings);
    }
    return site;
}

// One abstract socket name per user, which no file backs: the kernel frees it when its process exits, and refuses it to
// a second process while one holds it.
struct SocketAddress {
    sockaddr_un addr{};
    socklen_t size = 0;
};
SocketAddress socket_address() {
    SocketAddress address;
    address.addr.sun_family = AF_UNIX;
    const std::string name = "tt-metal-streaming-profiler-" + std::to_string(::getuid());
    std::memcpy(address.addr.sun_path + 1, name.data(), name.size());  // the leading zero byte makes it abstract
    address.size = static_cast<socklen_t>(offsetof(sockaddr_un, sun_path) + 1 + name.size());
    return address;
}

int connect_to_external_process() {
    const SocketAddress address = socket_address();
    const int fd = ::socket(AF_UNIX, SOCK_STREAM | SOCK_CLOEXEC, 0);
    TT_FATAL(fd >= 0, "external link: socket() failed: {}", std::strerror(errno));
    if (::connect(fd, reinterpret_cast<const sockaddr*>(&address.addr), address.size) != 0) {
        ::close(fd);
        return -1;
    }
    // An external process that stops reading must not hold up this process, whose image loads push call sites: a
    // send that waits this long ends the link instead.
    const timeval send_timeout{.tv_sec = 1, .tv_usec = 0};
    ::setsockopt(fd, SOL_SOCKET, SO_SNDTIMEO, &send_timeout, sizeof(send_timeout));
    return fd;
}

}  // namespace

ProfiledEnd::ProfiledEnd(const Hello& hello, Detach detach) :
    hello_(encode_hello(hello)), relay_count_(static_cast<uint32_t>(hello.relays.size())), detach_(std::move(detach)) {
    TT_FATAL(::pipe2(wake_, O_CLOEXEC | O_NONBLOCK) == 0, "external link: pipe2() failed: {}", std::strerror(errno));
}

void ProfiledEnd::start() {
    // An image's call sites go out while it loads, so the external process holds them before any of its records,
    // even if this process dies right after.
    tt::llrt::set_call_site_listener([this](uint32_t, uint32_t) {
        const std::lock_guard lock(send_mu_);
        if (!push_sites()) {
            ::shutdown(fd_, SHUT_RDWR);  // the link's thread sees it end
        }
    });
    thread_ = std::thread([this] { run(); });
}

ProfiledEnd::~ProfiledEnd() {
    tt::llrt::set_call_site_listener({});
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

void ProfiledEnd::wake() {
    const char byte = 0;
    [[maybe_unused]] const ssize_t written = ::write(wake_[1], &byte, 1);
}

void ProfiledEnd::run() {
    while (true) {
        if (std::lock_guard lock(mu_); stop_) {
            return;
        }
        if (const int fd = connect_to_external_process(); fd >= 0) {
            serve(fd);
        }
        pollfd waker{wake_[0], POLLIN, 0};
        if (::poll(&waker, 1, kReconnectMs) > 0) {
            char drain[64];
            while (::read(wake_[0], drain, sizeof(drain)) > 0) {
            }
        }
    }
}

bool ProfiledEnd::push_sites() {
    for (const uint32_t taken = tt::llrt::call_site_ids_taken(); fd_ >= 0 && sites_pushed_ < taken; sites_pushed_++) {
        const tt::llrt::CallSite* site = tt::llrt::find_call_site(sites_pushed_);
        if (site == nullptr) {
            break;  // its image is still being read; the image's own push sends it
        }
        if (is_external_site(*site)) {
            Writer w = message(kSitePush);
            w.u32(sites_pushed_);
            put_site(w, *site);
            if (!send_message(fd_, w.out)) {
                return false;
            }
        }
    }
    return true;
}

// Sends the hello and the call sites loaded so far, then holds the link until either side closes it.
void ProfiledEnd::serve(int fd) {
    std::unique_lock sending(send_mu_);
    fd_ = fd;
    sites_pushed_ = 0;
    bool open = send_message(fd, hello_);
    // Every call site whose records go to the external process that loaded before it connected.
    open = open && push_sites();
    sending.unlock();
    // The external process sends nothing, so its end turning readable means the link closed.
    std::array<pollfd, 2> fds{{{fd, POLLIN, 0}, {wake_[0], POLLIN, 0}}};
    while (open) {
        if (::poll(fds.data(), fds.size(), -1) < 0) {
            open = errno == EINTR;
            continue;
        }
        if (fds[0].revents != 0) {
            break;
        }
        char drain[64];
        while (::read(wake_[0], drain, sizeof(drain)) > 0) {
        }
        if (std::lock_guard lock(mu_); stop_) {
            break;
        }
    }
    {
        const std::lock_guard lock(send_mu_);
        fd_ = -1;
    }
    // The external process may have gone without turning the relays' copies off.
    for (uint32_t relay = 0; relay < relay_count_; relay++) {
        detach_(relay);
    }
    ::close(fd);
}

ExternalListener::ExternalListener() : fd_(::socket(AF_UNIX, SOCK_STREAM | SOCK_CLOEXEC, 0)) {}

ExternalListener::~ExternalListener() { ::close(fd_); }

bool ExternalListener::listen() {
    const SocketAddress address = socket_address();
    return ::bind(fd_, reinterpret_cast<const sockaddr*>(&address.addr), address.size) == 0 && ::listen(fd_, 4) == 0;
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

ExternalEnd::ExternalEnd(int fd) : fd_(fd) {
    ucred peer{};
    socklen_t size = sizeof(peer);
    if (::getsockopt(fd_, SOL_SOCKET, SO_PEERCRED, &peer, &size) == 0) {
        pid_ = peer.pid;
    }
    if (const auto first = recv_message(fd_)) {
        Reader r{*first};
        TT_FATAL(r.u32() == kHello, "external link: the profiled process sent no hello");
        hello_ = decode_hello(r);
    } else {
        closed_.store(true, std::memory_order_release);
    }
}

ExternalEnd::~ExternalEnd() {
    ::shutdown(fd_, SHUT_RDWR);
    if (thread_.joinable()) {
        thread_.join();
    }
    ::close(fd_);
}

void ExternalEnd::start() {
    thread_ = std::thread([this] { read_thread(); });
}

// A link that fails here ends, and this process goes on to serve the next profiled process.
void ExternalEnd::read_thread() {
    try {
        while (const auto received = recv_message(fd_)) {
            Reader r{*received};
            TT_FATAL(r.u32() == kSitePush, "external link: unknown message");
            const uint32_t id = r.u32();
            const uint32_t local = tt::llrt::add_call_site(decode_site(r));
            std::lock_guard lock(sites_mu_);
            sites_.try_emplace(id, local);
            sites_cv_.notify_all();
        }
    } catch (const std::exception& e) {
        log_error(tt::LogMetal, "[streaming profiler] dropping the link to Metal process {}: {}", pid_, e.what());
        drop();
    }
    std::lock_guard lock(sites_mu_);
    closed_.store(true, std::memory_order_release);
    sites_cv_.notify_all();
}

void ExternalEnd::drop() { ::shutdown(fd_, SHUT_RDWR); }

std::optional<uint32_t> ExternalEnd::local_id(uint32_t id) {
    std::unique_lock lock(sites_mu_);
    sites_cv_.wait(lock, [&] { return sites_.contains(id) || closed(); });
    const auto it = sites_.find(id);
    return it == sites_.end() ? std::nullopt : std::optional(it->second);
}

}  // namespace tt::tt_metal::streaming_profiler::external
