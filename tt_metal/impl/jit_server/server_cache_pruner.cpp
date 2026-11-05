// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "impl/jit_server/server_cache_pruner.hpp"

#include <signal.h>
#include <sys/stat.h>
#include <sys/statvfs.h>
#include <sys/types.h>
#include <unistd.h>

#include <algorithm>
#include <cerrno>
#include <cmath>
#include <cstdlib>
#include <ctime>
#include <optional>
#include <stdexcept>
#include <utility>
#include <string_view>
#include <system_error>

namespace tt::tt_metal::jit_server {

namespace fs = std::filesystem;

namespace {

constexpr std::string_view kTrashPrefix = ".jit-prune-trash.";
// Ten years: far beyond any sensible setting, and small enough that no time arithmetic overflows.
constexpr double kMaxSeconds = 10.0 * 365 * 24 * 3600;

double parse_number(const std::string& text, const char* name, double max) {
    std::size_t used = 0;
    double value = 0.0;
    try {
        value = std::stod(text, &used);
    } catch (const std::exception&) {
        used = 0;
    }
    if (used == 0 || used != text.size() || !std::isfinite(value) || value < 0.0 || value > max) {
        throw std::invalid_argument(
            std::string(name) + " must be a number from 0 to " + std::to_string(static_cast<long long>(max)) +
            ", got '" + text + "'");
    }
    return value;
}

std::optional<std::string> read_env(const CachePruneConfig::GetEnvFn& get_env, const char* name) {
    const char* value = get_env(name);
    if (value == nullptr || *value == '\0') {
        return std::nullopt;
    }
    return std::string(value);
}

fs::path normalize(const fs::path& path) {
    fs::path normal = path.lexically_normal();
    if (!normal.has_filename() && normal.has_parent_path() && normal != normal.root_path()) {
        normal = normal.parent_path();
    }
    return normal;
}

bool is_trash(const fs::path& path) { return path.filename().string().rfind(kTrashPrefix, 0) == 0; }

bool is_build_key(const fs::path& path) {
    const std::string name = path.filename().string();
    return !name.empty() && std::all_of(name.begin(), name.end(), [](unsigned char c) { return c >= '0' && c <= '9'; });
}

// The pid embedded in a trash name: .jit-prune-trash.<variant>.<pid>.<counter>
std::optional<pid_t> trash_owner(const fs::path& path) {
    const std::string name = path.filename().string();
    const auto last = name.rfind('.');
    if (last == std::string::npos || last == 0) {
        return std::nullopt;
    }
    const auto previous = name.rfind('.', last - 1);
    if (previous == std::string::npos) {
        return std::nullopt;
    }
    const std::string pid = name.substr(previous + 1, last - previous - 1);
    if (pid.empty() || pid.size() > 9 ||
        !std::all_of(pid.begin(), pid.end(), [](unsigned char c) { return c >= '0' && c <= '9'; })) {
        return std::nullopt;
    }
    return static_cast<pid_t>(std::stol(pid));
}

bool process_alive(pid_t pid) { return pid > 0 && (::kill(pid, 0) == 0 || errno == EPERM); }

// Trash this process made (a failed delete to retry) or that a process which has since exited
// left behind. Another live server's trash is its own to finish.
bool trash_is_ours_to_remove(const fs::path& path) {
    const auto owner = trash_owner(path);
    return !owner || *owner == ::getpid() || !process_alive(*owner);
}

bool same_directory(const fs::path& a, const fs::path& b) {
    std::error_code ec;
    if (fs::equivalent(a, b, ec)) {
        return true;  // also catches symlinks and bind mounts when both exist
    }
    return normalize(fs::absolute(a, ec)) == normalize(fs::absolute(b, ec));
}

double free_pct(const DiskSpace& space) {
    return space.capacity == 0 ? 100.0 : 100.0 * static_cast<double>(space.available) / space.capacity;
}

// Real subdirectories of path, not following symlinks. Entries that vanish mid-scan are skipped.
std::vector<fs::path> child_dirs(const fs::path& path) {
    std::vector<fs::path> dirs;
    std::error_code ec;
    fs::directory_iterator it(path, ec);
    for (; !ec && it != fs::directory_iterator(); it.increment(ec)) {
        std::error_code type_ec;
        if (it->is_directory(type_ec) && !it->is_symlink(type_ec)) {
            dirs.push_back(it->path());
        }
    }
    return dirs;
}

struct Usage {
    std::int64_t last_used = 0;
    std::uint64_t bytes = 0;
};

// Newest timestamp and allocated size of a directory tree. The directory mtimes count, and so do
// each file's mtime and atime; directory atimes are skipped because the scan itself updates them.
Usage measure(const fs::path& path) {
    Usage usage;
    struct stat info{};
    if (::lstat(path.c_str(), &info) != 0) {
        return usage;
    }
    usage.bytes = static_cast<std::uint64_t>(info.st_blocks) * 512;
    usage.last_used = info.st_mtime;
    if (!S_ISDIR(info.st_mode)) {
        usage.last_used = std::max<std::int64_t>(info.st_mtime, info.st_atime);
        return usage;
    }
    std::error_code ec;
    fs::directory_iterator it(path, ec);
    for (; !ec && it != fs::directory_iterator(); it.increment(ec)) {
        const Usage child = measure(it->path());
        usage.bytes += child.bytes;
        usage.last_used = std::max(usage.last_used, child.last_used);
    }
    return usage;
}

std::string megabytes(std::uint64_t bytes) { return std::to_string(bytes / (1024 * 1024)); }

std::string percent(double value) {
    const auto tenths = static_cast<long long>(value * 10.0 + 0.5);
    return std::to_string(tenths / 10) + "." + std::to_string(tenths % 10) + "%";
}

}  // namespace

CachePruneConfig CachePruneConfig::from_env(const GetEnvFn& get_env) {
    CachePruneConfig config;
    if (auto value = read_env(get_env, kCacheMinFreePctEnv)) {
        config.start_below_free_pct = parse_number(*value, kCacheMinFreePctEnv, 100.0);
    }
    std::optional<double> target;
    if (auto value = read_env(get_env, kCacheTargetFreePctEnv)) {
        target = parse_number(*value, kCacheTargetFreePctEnv, 100.0);
    }
    if (auto value = read_env(get_env, kCacheMinIdleEnv)) {
        config.min_idle =
            std::chrono::seconds(static_cast<std::int64_t>(parse_number(*value, kCacheMinIdleEnv, kMaxSeconds)));
    }
    if (auto value = read_env(get_env, kCacheCheckIntervalEnv)) {
        config.check_interval =
            std::chrono::seconds(static_cast<std::int64_t>(parse_number(*value, kCacheCheckIntervalEnv, kMaxSeconds)));
        if (config.check_interval.count() < 1) {
            throw std::invalid_argument(std::string(kCacheCheckIntervalEnv) + " must be at least 1");
        }
    }
    if (!config.enabled()) {
        for (const char* name : {kCacheTargetFreePctEnv, kCacheMinIdleEnv, kCacheCheckIntervalEnv}) {
            if (read_env(get_env, name)) {
                config.ignored_settings.emplace_back(name);
            }
        }
        return config;
    }
    config.stop_at_free_pct = target.value_or(std::min(config.start_below_free_pct + 10.0, 100.0));
    if (config.stop_at_free_pct < config.start_below_free_pct) {
        throw std::invalid_argument(std::string(kCacheTargetFreePctEnv) + " must not be below " + kCacheMinFreePctEnv);
    }
    return config;
}

CachePruneConfig CachePruneConfig::from_env() {
    return from_env([](const char* name) -> const char* { return std::getenv(name); });
}

void CachePruneConfig::validate_cache_root(const fs::path& root, const GetEnvFn& get_env) const {
    if (!enabled()) {
        return;
    }
    if (!read_env(get_env, kServerCacheRootEnv)) {
        throw std::invalid_argument(
            std::string("Cache pruning needs ") + kServerCacheRootEnv + " set to a directory only this server uses. " +
            "The default, " + kDefaultServerCacheRoot +
            ", is also where a client with no $HOME keeps its kernel cache, and pruning would delete it.");
    }
    std::vector<std::pair<fs::path, std::string>> client_caches = {
        {fs::path(kDefaultServerCacheRoot), "the client kernel cache used when $HOME is unset"}};
    if (auto home = read_env(get_env, "HOME")) {
        client_caches.emplace_back(fs::path(*home) / ".cache" / "tt-metal-cache", "the default client kernel cache");
    }
    if (auto cache = read_env(get_env, "TT_METAL_CACHE")) {
        client_caches.emplace_back(
            fs::path(*cache) / "tt-metal-cache", "the client kernel cache TT_METAL_CACHE selects");
    }
    for (const auto& [client_cache, description] : client_caches) {
        if (same_directory(root, client_cache)) {
            throw std::invalid_argument(
                std::string(kServerCacheRootEnv) + "=" + root.string() + " is " + description +
                ", which has the same layout, so pruning would delete ELFs clients are loading. Use a directory "
                "only this server uses.");
        }
    }
}

void CachePruneConfig::validate_cache_root(const fs::path& root) const {
    validate_cache_root(root, [](const char* name) -> const char* { return std::getenv(name); });
}

std::chrono::seconds next_prune_delay(
    std::chrono::seconds previous, const CachePruneResult& result, const CachePruneConfig& config) {
    if (result.ran && result.freed_bytes == 0 && !result.interrupted) {
        const std::chrono::seconds cap = std::max(kMaxPruneBackoff, config.check_interval);
        return std::min(std::max(previous, config.check_interval) * 2, cap);
    }
    return config.check_interval;
}

DiskSpace disk_space_from_counts(
    std::uint64_t blocks, std::uint64_t free_blocks, std::uint64_t available_blocks, std::uint64_t block_size) {
    // Like df: used space plus what an unprivileged user may still allocate.
    const std::uint64_t used = blocks > free_blocks ? blocks - free_blocks : 0;
    return {(used + available_blocks) * block_size, available_blocks * block_size};
}

DiskSpace statvfs_disk_space(const fs::path& path) {
    struct statvfs stats{};
    if (::statvfs(path.c_str(), &stats) != 0) {
        throw std::system_error(errno, std::generic_category(), "statvfs " + path.string());
    }
    return disk_space_from_counts(stats.f_blocks, stats.f_bfree, stats.f_bavail, stats.f_frsize);
}

ServerCachePruner::ServerCachePruner(fs::path cache_root, CachePruneConfig config, LogFn log, DiskSpaceFn disk_space) :
    root_(normalize(cache_root)), config_(config), log_(std::move(log)), disk_space_(std::move(disk_space)) {}

ServerCachePruner::~ServerCachePruner() { stop(); }

ServerCachePruner::Lease ServerCachePruner::acquire(const fs::path& kernel_dir) {
    if (!config_.enabled()) {
        return Lease();
    }
    std::string key = normalize(kernel_dir).string();
    {
        std::lock_guard lock(mutex_);
        ++in_use_[key];
    }
    // Record the use. A directory that does not exist yet gets a fresh mtime when it is created.
    std::error_code ec;
    fs::last_write_time(key, fs::file_time_type::clock::now(), ec);
    return Lease(this, std::move(key));
}

void ServerCachePruner::release(const std::string& key) {
    std::lock_guard lock(mutex_);
    auto it = in_use_.find(key);
    if (it != in_use_.end() && --it->second == 0) {
        in_use_.erase(it);
    }
}

void ServerCachePruner::start() {
    if (!config_.enabled() || worker_.joinable()) {
        return;
    }
    {
        std::lock_guard lock(stop_mutex_);
        stopping_ = false;
    }
    worker_ = std::thread([this] { run(); });
}

void ServerCachePruner::stop() {
    {
        std::lock_guard lock(stop_mutex_);
        stopping_ = true;
    }
    stop_cv_.notify_all();
    if (worker_.joinable()) {
        worker_.join();
    }
}

bool ServerCachePruner::idle_enough(std::int64_t last_used, std::int64_t now) const {
    return now - last_used >= static_cast<std::int64_t>(config_.min_idle.count());
}

bool ServerCachePruner::in_use_locked(const fs::path& path) const {
    // A lease on this directory or on one that contains it...
    for (fs::path dir = path; !dir.empty(); dir = dir.parent_path()) {
        if (in_use_.count(dir.string()) > 0) {
            return true;
        }
        if (dir == dir.parent_path()) {
            break;
        }
    }
    // ...or on anything inside it: those keys sort right after "<path>/".
    const std::string prefix = path.string() + "/";
    const auto inside = in_use_.lower_bound(prefix);
    return inside != in_use_.end() && inside->first.compare(0, prefix.size(), prefix) == 0;
}

ServerCachePruner::ScanResult ServerCachePruner::scan(std::int64_t now) const {
    ScanResult result;
    for (const auto& build_key : child_dirs(root_)) {
        if (!is_build_key(build_key)) {
            continue;  // e.g. the shared pch/ directory
        }
        for (const auto& kernel : child_dirs(build_key / "kernels")) {
            if (stopping()) {
                return result;
            }
            for (const auto& variant : child_dirs(kernel)) {
                if (is_trash(variant)) {
                    if (trash_is_ours_to_remove(variant)) {
                        result.trash.push_back(variant);
                    }
                    continue;
                }
                const Usage usage = measure(variant);
                if (usage.bytes > 0 && idle_enough(usage.last_used, now)) {
                    result.candidates.push_back({usage.last_used, usage.bytes, variant});
                }
            }
        }
    }
    return result;
}

ServerCachePruner::RemoveOutcome ServerCachePruner::remove(
    const Candidate& candidate, std::int64_t now, std::uint64_t& freed, std::string& error) {
    freed = 0;
    std::error_code ec;
    fs::path trash;
    {
        std::lock_guard lock(mutex_);
        if (in_use_locked(candidate.path)) {
            return RemoveOutcome::InUse;
        }
        // Every lease stamps the directory's own mtime, so one lstat shows any use since the scan
        // without walking the tree while compile requests wait on this mutex.
        struct stat info{};
        if (::lstat(candidate.path.c_str(), &info) != 0) {
            if (errno == ENOENT) {
                return RemoveOutcome::Gone;  // someone else deleted it since the scan
            }
            error = "stat " + candidate.path.string() + ": " + std::generic_category().message(errno);
            return RemoveOutcome::Failed;
        }
        if (!idle_enough(info.st_mtime, now)) {
            return RemoveOutcome::UsedSinceScan;
        }
        // Renaming within the same directory needs no new disk blocks, so it still works on a full
        // disk, and it takes the directory out of every request's reach at once.
        trash = candidate.path.parent_path() / (std::string(kTrashPrefix) + candidate.path.filename().string() + "." +
                                                std::to_string(::getpid()) + "." + std::to_string(++trash_counter_));
        fs::rename(candidate.path, trash, ec);
        if (ec == std::errc::no_such_file_or_directory) {
            return RemoveOutcome::Gone;
        }
        if (ec) {
            error = "rename " + candidate.path.string() + ": " + ec.message();
            return RemoveOutcome::Failed;
        }
    }

    fs::remove_all(trash, ec);
    RemoveOutcome outcome = RemoveOutcome::Deleted;
    freed = candidate.bytes;
    if (ec) {
        // Count only what actually went. The rest stays as trash for the next run to retry.
        const std::uint64_t remaining = measure(trash).bytes;
        freed = candidate.bytes > remaining ? candidate.bytes - remaining : 0;
        error = "delete " + trash.string() + ": " + ec.message();
        outcome = RemoveOutcome::Failed;
    }
    {
        // Drop the kernel's own directory once its last variant is gone. remove() only deletes an
        // empty directory, and the lease check stops it racing a request that is creating one.
        std::lock_guard lock(mutex_);
        const fs::path kernel = candidate.path.parent_path();
        if (!in_use_locked(kernel)) {
            fs::remove(kernel, ec);
        }
    }
    return outcome;
}

CachePruneResult ServerCachePruner::prune_once() {
    std::lock_guard prune_lock(prune_mutex_);
    CachePruneResult result;
    std::error_code ec;
    if (!config_.enabled() || !fs::is_directory(root_, ec)) {
        return result;
    }
    DiskSpace space = disk_space_(root_);
    result.free_pct_before = result.free_pct_after = free_pct(space);
    if (result.free_pct_before >= config_.start_below_free_pct) {
        return result;
    }
    result.ran = true;

    // Re-measured as the run goes: concurrent compiles keep writing, and a file still open elsewhere
    // frees nothing when deleted, so the sizes measured by the scan are only an estimate.
    auto remeasure = [&](std::uint64_t estimated_freed) {
        try {
            space = disk_space_(root_);
        } catch (const std::exception&) {
            space.available += estimated_freed;
        }
    };
    const auto target =
        static_cast<std::uint64_t>(static_cast<long double>(space.capacity) * config_.stop_at_free_pct / 100.0L);
    auto still_needed = [&]() -> std::uint64_t { return target > space.available ? target - space.available : 0; };

    const std::int64_t now = std::time(nullptr);
    ScanResult scanned = scan(now);

    // Trash an earlier failed or interrupted delete left behind goes before anything live.
    if (!scanned.trash.empty()) {
        std::uint64_t trash_bytes = 0;
        for (const auto& trash : scanned.trash) {
            trash_bytes += measure(trash).bytes;
            fs::remove_all(trash, ec);
        }
        remeasure(trash_bytes);
    }

    std::sort(scanned.candidates.begin(), scanned.candidates.end(), [](const Candidate& a, const Candidate& b) {
        return a.last_used < b.last_used;
    });
    // Filesystems that free blocks lazily (btrfs, ZFS) can lag behind the deletes; stop at twice the
    // initial shortfall rather than delete the whole cache waiting for them to catch up.
    const std::uint64_t initial_need = still_needed();
    const std::uint64_t deletion_cap = initial_need > UINT64_MAX / 2 ? UINT64_MAX : initial_need * 2;

    for (const auto& candidate : scanned.candidates) {
        if (still_needed() == 0 || result.freed_bytes >= deletion_cap || stopping()) {
            break;
        }
        std::uint64_t freed = 0;
        std::string error;
        switch (remove(candidate, now, freed, error)) {
            case RemoveOutcome::Deleted: ++result.deleted; break;
            case RemoveOutcome::Gone: break;
            case RemoveOutcome::InUse: ++result.skipped_in_use; break;
            case RemoveOutcome::UsedSinceScan: ++result.skipped_used_since_scan; break;
            case RemoveOutcome::Failed:
                ++result.failed;
                if (result.first_error.empty()) {
                    result.first_error = error;
                }
                break;
        }
        if (freed > 0) {
            result.freed_bytes += freed;
            remeasure(freed);
        }
    }
    result.interrupted = stopping();
    result.free_pct_after = free_pct(space);
    result.still_needed_bytes = still_needed();
    result.reached_target = result.still_needed_bytes == 0;
    log(summary(result));
    return result;
}

void ServerCachePruner::run() {
    std::chrono::seconds delay = config_.check_interval;
    while (!stopping()) {
        CachePruneResult result;
        try {
            result = prune_once();
        } catch (const std::exception& e) {
            log(std::string("Cache prune failed: ") + e.what());
        }
        delay = next_prune_delay(delay, result, config_);
        if (delay > config_.check_interval) {
            log("Cache prune: freed nothing; next check in " + std::to_string(delay.count()) + " s");
        }
        std::unique_lock lock(stop_mutex_);
        if (stop_cv_.wait_for(lock, delay, [this] { return stopping(); })) {
            return;
        }
    }
}

void ServerCachePruner::log(const std::string& message) const {
    if (!log_) {
        return;
    }
    try {
        log_(message);
    } catch (...) {
        // A failed log write (e.g. a full disk under a redirected log) must not stop pruning.
    }
}

std::string ServerCachePruner::summary(const CachePruneResult& result) const {
    std::string text = "Cache prune: free space " + percent(result.free_pct_before) + " < " +
                       percent(config_.start_below_free_pct) + "; deleted " + std::to_string(result.deleted) +
                       " kernel directories (" + megabytes(result.freed_bytes) + " MB); free space now " +
                       percent(result.free_pct_after);
    std::vector<std::string> skipped;
    if (result.skipped_in_use > 0) {
        skipped.push_back(std::to_string(result.skipped_in_use) + " in use by a request");
    }
    if (result.skipped_used_since_scan > 0) {
        skipped.push_back(std::to_string(result.skipped_used_since_scan) + " used during the prune");
    }
    if (result.failed > 0) {
        skipped.push_back(
            std::to_string(result.failed) + " could not be deleted (first error: " + result.first_error + ")");
    }
    if (!skipped.empty()) {
        text += "; skipped ";
        for (std::size_t i = 0; i < skipped.size(); ++i) {
            text += (i == 0 ? "" : ", ") + skipped[i];
        }
    }
    if (result.interrupted) {
        text += "; stopped early because the server is shutting down";
    } else if (!result.reached_target) {
        text += ". Could not reach " + percent(config_.stop_at_free_pct) + ": " + megabytes(result.still_needed_bytes) +
                " MB more is needed";
        if (skipped.empty()) {
            text += ", but nothing else has been idle for " + std::to_string(config_.min_idle.count()) +
                    " s (firmware and PCH are never deleted)";
        }
    }
    return text;
}

}  // namespace tt::tt_metal::jit_server
