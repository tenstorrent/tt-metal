// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "tools/profiler/streaming_profiler_consumer.hpp"

#include <algorithm>
#include <cstdlib>
#include <mutex>

#include <fmt/format.h>
#include <tt-logger/tt-logger.hpp>
#include <tt_stl/assert.hpp>
#include <tt_stl/indestructible.hpp>

#include "hostdevcommon/profiler_zone_id.h"
#include "llrt/zone_meta.hpp"
#include "tools/profiler/streaming_profiler_receiver.hpp"

namespace tt::tt_metal::streaming_profiler {

void ZoneNameMirror::refresh() {
    std::vector<llrt::ZoneMetaEntry> delta;
    cursor_ = llrt::ZoneMetaRegistry::instance().additions_since(cursor_, delta);
    for (auto& e : delta) {
        names_.emplace(e.zone_id, std::move(e.name));
    }
}

void log_unnamed_ids(std::string_view consumer_name, const ZoneNameMirror& mirror) {
    if (mirror.unnamed() == 0) {
        return;
    }
    std::string ids;
    for (uint32_t id : mirror.unnamed_ids()) {
        ids +=
            fmt::format("{}{} (tu {} local {})", ids.empty() ? "" : ", ", id, TT_ZONE_TU_OF(id), TT_ZONE_LOCAL_OF(id));
    }
    log_warning(
        tt::LogMetal,
        "[streaming profiler receiver] consumer \"{}\": {} unnamed marker rows [MUST be 0 -- a binary loaded without "
        ".tt_zone_meta, or two TUs share a tu_id]; ids (up to {} distinct): {}",
        consumer_name,
        mirror.unnamed(),
        ZoneNameMirror::kMaxUnnamedIds,
        ids);
}

namespace {

struct Registry {
    struct Entry {
        StreamingProfilerConsumerHandle handle = 0;
        std::string name;
        StreamingProfilerRecordCallback cb;
        StreamingProfilerConsumerHandle live = 0;  // receiver-side handle while a capture is active
    };
    std::mutex mu;
    std::vector<Entry> entries;
    StreamingProfilerConsumerHandle next_handle = 1;
    StreamingProfilerReceiver* receiver = nullptr;
};

Registry& registry() {
    static ttsl::Indestructible<Registry> r;
    return r.get();
}

struct FileConsumer {
    std::string name;
    std::string (*path_getter)();
    StreamingProfilerRecordCallback on_batch;
    std::function<void(const std::string&)> write;
    std::string path;
    StreamingProfilerConsumerHandle handle = 0;
    bool resolved = false;
};

std::vector<FileConsumer*>& file_consumers() {
    static ttsl::Indestructible<std::vector<FileConsumer*>> v;
    return v.get();
}

// Ask each declared file consumer for its path and register the enabled ones. Runs at capture attach,
// not at declaration, because the paths come from rtoptions.
void resolve_file_consumers() {
    static bool exit_hooked = false;
    for (FileConsumer* fc : file_consumers()) {
        if (fc->resolved) {
            continue;
        }
        fc->resolved = true;
        fc->path = fc->path_getter();
        if (fc->path.empty()) {
            continue;
        }
        fc->handle = register_consumer(fc->name, fc->on_batch);
        if (!exit_hooked) {
            exit_hooked = true;
            // Runs after receiver shutdown has delivered every buffered batch, and unregisters first so
            // no batch can race the write.
            std::atexit([] {
                for (FileConsumer* c : file_consumers()) {
                    if (c->handle != 0) {
                        unregister_consumer(c->handle);
                        c->write(c->path);
                    }
                }
            });
        }
    }
}

}  // namespace

StreamingProfilerConsumerHandle register_consumer(std::string name, StreamingProfilerRecordCallback cb) {
    Registry& r = registry();
    std::lock_guard<std::mutex> lk(r.mu);
    Registry::Entry e{r.next_handle++, std::move(name), std::move(cb)};
    if (r.receiver != nullptr) {
        e.live = r.receiver->add_consumer(e.name, e.cb);
    }
    const StreamingProfilerConsumerHandle handle = e.handle;
    r.entries.push_back(std::move(e));
    return handle;
}

void unregister_consumer(StreamingProfilerConsumerHandle handle) {
    Registry& r = registry();
    std::lock_guard<std::mutex> lk(r.mu);
    auto it = std::find_if(r.entries.begin(), r.entries.end(), [&](const auto& e) { return e.handle == handle; });
    TT_FATAL(it != r.entries.end(), "unknown streaming profiler consumer handle {}", handle);
    if (it->live != 0 && r.receiver != nullptr) {
        r.receiver->remove_consumer(it->live);
    }
    r.entries.erase(it);
}

void register_file_consumer_impl(
    std::string name,
    std::string (*path)(),
    StreamingProfilerRecordCallback on_batch,
    std::function<void(const std::string&)> write) {
    file_consumers().push_back(new FileConsumer{std::move(name), path, std::move(on_batch), std::move(write)});
}

void attach_registered_consumers(StreamingProfilerReceiver& receiver) {
    resolve_file_consumers();  // registers through register_consumer(), so it must run before the lock
    Registry& r = registry();
    std::lock_guard<std::mutex> lk(r.mu);
    r.receiver = &receiver;
    for (auto& e : r.entries) {
        e.live = receiver.add_consumer(e.name, e.cb);
    }
}

void detach_registered_consumers() {
    Registry& r = registry();
    std::lock_guard<std::mutex> lk(r.mu);
    r.receiver = nullptr;
    for (auto& e : r.entries) {
        e.live = 0;
    }
}

}  // namespace tt::tt_metal::streaming_profiler
