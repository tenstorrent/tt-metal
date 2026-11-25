// SPDX-FileCopyrightText: © 2026 Olof Johansson <olof@lixom.net>
//
// SPDX-License-Identifier: Apache-2.0

#include "kernel_cache_eviction.hpp"

#include <fcntl.h>
#include <sys/file.h>
#include <sys/stat.h>
#include <unistd.h>

#include <algorithm>
#include <atomic>
#include <cctype>
#include <cmath>
#include <cstdio>
#include <charconv>
#include <cerrno>
#include <cstring>
#include <ctime>
#include <fstream>
#include <functional>
#include <limits>
#include <mutex>
#include <string>
#include <system_error>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <vector>

#include <fmt/format.h>
#include <tt-logger/tt-logger.hpp>

namespace fs = std::filesystem;

namespace tt::tt_metal {
namespace {

constexpr const char* kLockName = ".trim.lock";
constexpr const char* kTrashName = ".trash";
constexpr const char* kPchName = "pch";
constexpr const char* kStatsName = ".stats";

// A full scan at least this often fixes drift in the size counter (deletes by hand, crashed writers).
constexpr std::time_t kResyncInterval = 24 * 60 * 60;

std::string normalize(const fs::path& path) {
    std::string s = path.lexically_normal().string();
    while (s.size() > 1 && s.back() == '/') {
        s.pop_back();
    }
    return s;
}

// The cache this process keeps under its cap (set_kernel_cache_cap), and the directories it stamped as used,
// with the time of its last stamp. Inactive, and empty, without a cap.
struct Upkeep {
    std::atomic<bool> enabled{false};
    std::mutex mutex;
    fs::path root;
    std::uint64_t max_bytes = 0;
    std::chrono::seconds min_evict_age{0};
    std::unordered_map<std::string, std::time_t> last_stamp;

    static Upkeep& inst() {
        static Upkeep instance;
        return instance;
    }
};

struct InodeKey {
    dev_t dev;
    ino_t ino;
    bool operator==(const InodeKey&) const = default;
};

struct InodeKeyHash {
    std::size_t operator()(const InodeKey& k) const noexcept {
        return std::hash<std::uint64_t>{}(
            (static_cast<std::uint64_t>(k.dev) << 32) ^ static_cast<std::uint64_t>(k.ino));
    }
};

using SeenInodes = std::unordered_set<InodeKey, InodeKeyHash>;

// Disk usage of path and everything below it, like du: allocated blocks, a hard-linked file counted once.
// Entries that vanish while we walk (another process evicting or renaming) simply stop counting.
std::uint64_t disk_usage(const fs::path& path, SeenInodes& seen) {
    struct stat st{};
    if (::lstat(path.c_str(), &st) != 0) {
        return 0;
    }
    if (!S_ISDIR(st.st_mode) && st.st_nlink > 1 && !seen.insert({st.st_dev, st.st_ino}).second) {
        return 0;
    }
    std::uint64_t size = static_cast<std::uint64_t>(st.st_blocks) * 512;
    if (S_ISDIR(st.st_mode)) {
        std::error_code ec;
        for (fs::directory_iterator it(path, ec), end; !ec && it != end; it.increment(ec)) {
            size += disk_usage(it->path(), seen);
        }
    }
    return size;
}

// Size of the directory entry itself, without its children.
std::uint64_t own_usage(const fs::path& path) {
    struct stat st{};
    return ::lstat(path.c_str(), &st) == 0 ? static_cast<std::uint64_t>(st.st_blocks) * 512 : 0;
}

std::time_t mtime_of(const fs::path& path) {
    struct stat st{};
    return ::lstat(path.c_str(), &st) == 0 ? st.st_mtime : 0;
}

bool is_dir(const fs::path& path) {
    std::error_code ec;
    return fs::is_directory(fs::symlink_status(path, ec));
}

// The root itself may be a symlink (e.g. ~/.cache on another disk); entries below it are never followed.
bool is_root_dir(const fs::path& root) {
    std::error_code ec;
    return fs::is_directory(root, ec);
}

std::vector<fs::path> list_dir(const fs::path& path) {
    std::vector<fs::path> out;
    std::error_code ec;
    for (fs::directory_iterator it(path, ec), end; !ec && it != end; it.increment(ec)) {
        out.push_back(it->path());
    }
    return out;
}

// An evictable entry: a kernel variant directory <build_key>/kernels/<kernel>/<variant>/ or a PCH directory
// pch/<hash>/.
struct Variant {
    fs::path path;
    std::time_t mtime;
    std::uint64_t size;
};

// Splits the cache into evictable entries and the size of everything else (firmware, a .trash a trimmer could
// not delete, the config and lock files, ...).
std::pair<std::vector<Variant>, std::uint64_t> scan(const fs::path& root) {
    SeenInodes seen;
    std::vector<Variant> variants;
    std::uint64_t other = own_usage(root);
    const auto add_entries = [&](const fs::path& dir) {
        other += own_usage(dir);
        for (const auto& entry : list_dir(dir)) {
            if (!is_dir(entry)) {
                other += disk_usage(entry, seen);
                continue;
            }
            variants.push_back({entry, mtime_of(entry), disk_usage(entry, seen)});
        }
    };
    for (const auto& top : list_dir(root)) {
        if (top.filename() == kPchName && is_dir(top)) {
            add_entries(top);
            continue;
        }
        const fs::path kernels = top / "kernels";
        if (!is_dir(top) || !is_dir(kernels)) {
            other += disk_usage(top, seen);
            continue;
        }
        other += own_usage(top);
        for (const auto& child : list_dir(top)) {
            if (child.filename() != "kernels") {
                other += disk_usage(child, seen);
            }
        }
        other += own_usage(kernels);
        for (const auto& kernel : list_dir(kernels)) {
            if (!is_dir(kernel)) {
                other += disk_usage(kernel, seen);
                continue;
            }
            add_entries(kernel);
        }
    }
    return {std::move(variants), other};
}

std::string human_size(std::uint64_t bytes) {
    constexpr double kGiB = 1024.0 * 1024.0 * 1024.0;
    constexpr double kMiB = 1024.0 * 1024.0;
    if (bytes >= kGiB) {
        return fmt::format("{:.1f} GiB", bytes / kGiB);
    }
    return fmt::format("{:.1f} MiB", bytes / kMiB);
}

// Holds an exclusive flock on <root>/.trim.lock, which guards trims and the size counter. Non-blocking for a
// trim (another trimmer means skip), blocking for a counter update. The kernel drops it if the process dies.
class TrimLock {
public:
    explicit TrimLock(const fs::path& root, bool wait = false) {
        fd_ = ::open((root / kLockName).c_str(), O_RDWR | O_CREAT | O_CLOEXEC, 0644);
        if (fd_ >= 0 && ::flock(fd_, wait ? LOCK_EX : LOCK_EX | LOCK_NB) != 0) {
            ::close(fd_);
            fd_ = -1;
        }
    }
    ~TrimLock() {
        if (fd_ >= 0) {
            ::close(fd_);
        }
    }
    TrimLock(const TrimLock&) = delete;
    TrimLock& operator=(const TrimLock&) = delete;
    bool held() const { return fd_ >= 0; }

private:
    int fd_ = -1;
};

// <root>/.stats: the size counter, ccache style. size is the cache size in bytes as of the last update;
// scan_time and scan_size are when the last full scan ran and the size it left. Written whole (temp file,
// then rename) under the trim lock, so a reader sees an old or a new file, never a torn one.
struct Stats {
    std::uint64_t size = 0;
    std::time_t scan_time = 0;
    std::uint64_t scan_size = 0;
};

std::optional<Stats> read_stats(const fs::path& root) {
    std::ifstream in(root / kStatsName);
    std::string key1, eq1, key2, eq2, key3, eq3;
    Stats stats;
    long long scan_time = 0;
    if (!(in >> key1 >> eq1 >> stats.size >> key2 >> eq2 >> scan_time >> key3 >> eq3 >> stats.scan_size) ||
        key1 != "size" || key2 != "scan_time" || key3 != "scan_size" || eq1 != "=" || eq2 != "=" || eq3 != "=" ||
        scan_time <= 0) {
        return std::nullopt;  // missing or corrupt: the caller rescans
    }
    stats.scan_time = static_cast<std::time_t>(scan_time);
    return stats;
}

// Call with the trim lock held.
void write_stats(const fs::path& root, const Stats& stats) {
    const fs::path path = root / kStatsName;
    const fs::path tmp = root / fmt::format("{}.{}", kStatsName, ::getpid());
    {
        std::ofstream out(tmp);
        out << "size = " << stats.size << "\nscan_time = " << static_cast<long long>(stats.scan_time)
            << "\nscan_size = " << stats.scan_size << "\n";
        out.close();
        if (!out) {
            std::error_code ec;
            fs::remove(tmp, ec);
            return;  // no counter: the next process rescans
        }
    }
    std::error_code ec;
    fs::rename(tmp, path, ec);
    if (ec) {
        fs::remove(tmp, ec);
    }
}

// Whether the counter calls for a full scan now: once a day to fix drift, and whenever the cache is over the
// cap, except shortly after a scan that could not get under it (everything left was in use or never
// evictable; scanning again before min_evict_age passes would find the same).
bool scan_due(const Stats& stats, std::uint64_t max_bytes, std::chrono::seconds min_evict_age, std::time_t now) {
    if (now - stats.scan_time >= kResyncInterval) {
        return true;
    }
    if (stats.size <= max_bytes) {
        return false;
    }
    return stats.scan_size <= max_bytes || now - stats.scan_time >= static_cast<std::time_t>(min_evict_age.count());
}

}  // namespace

std::optional<std::uint64_t> parse_cache_size(std::string_view value) {
    // Number: digits with an optional fraction ("1.5", ".5", "5.").
    std::size_t pos = 0;
    long double number = 0;
    bool any_digit = false;
    for (; pos < value.size() && std::isdigit(static_cast<unsigned char>(value[pos])); pos++) {
        number = number * 10 + (value[pos] - '0');
        any_digit = true;
    }
    if (pos < value.size() && value[pos] == '.') {
        long double scale = 1;
        for (pos++; pos < value.size() && std::isdigit(static_cast<unsigned char>(value[pos])); pos++) {
            scale /= 10;
            number += (value[pos] - '0') * scale;
            any_digit = true;
        }
    }
    if (!any_digit) {
        return std::nullopt;
    }
    while (pos < value.size() && value[pos] == ' ') {
        pos++;
    }
    // Unit: k/K, M, G, T are powers of 1000, with an "i" powers of 1024, optionally followed by "B".
    // No unit means GiB.
    std::string_view unit = value.substr(pos);
    long double multiplier = 1024.0L * 1024 * 1024;
    if (!unit.empty()) {
        constexpr std::string_view prefixes = "kMGT";
        const char prefix = unit[0] == 'K' ? 'k' : unit[0];
        const auto index = prefixes.find(prefix);
        if (index == std::string_view::npos) {
            return std::nullopt;
        }
        unit.remove_prefix(1);
        long double base = 1000;
        if (!unit.empty() && unit[0] == 'i') {
            base = 1024;
            unit.remove_prefix(1);
        }
        if (!unit.empty() && unit[0] == 'B') {
            unit.remove_prefix(1);
        }
        if (!unit.empty()) {
            return std::nullopt;
        }
        multiplier = 1;
        for (std::size_t i = 0; i <= index; i++) {
            multiplier *= base;
        }
    }
    const long double bytes = number * multiplier;
    if (bytes >= static_cast<long double>(std::numeric_limits<std::uint64_t>::max())) {
        return std::nullopt;
    }
    return static_cast<std::uint64_t>(std::roundl(bytes));
}

namespace {

// A cap below this is almost certainly a typo, and would evict the whole cache on every start.
constexpr std::uint64_t kMinCap = 100'000'000;

// Parses a cap: a size that is 0 (no cap) or at least kMinCap. Warns and returns nullopt otherwise.
std::optional<std::uint64_t> parse_cap(std::string_view value, std::string_view where) {
    const auto size = parse_cache_size(value);
    if (!size) {
        log_warning(
            tt::LogBuildKernels, "{}: '{}' is not a size (e.g. 20G, 1.5G, 500M, 1Gi); ignoring it", where, value);
        return std::nullopt;
    }
    if (*size != 0 && *size < kMinCap) {
        log_warning(
            tt::LogBuildKernels,
            "{}: '{}' is below the smallest kernel cache cap of 100M; ignoring it (a bare number means GiB)",
            where,
            value);
        return std::nullopt;
    }
    return size;
}

struct ConfigValue {
    std::uint64_t bytes = 0;  // 0: no cap (no valid max_size line, or max_size = 0)
    std::string text;         // the max_size used, as written; empty without a valid max_size line
    bool invalid = false;     // the file is unreadable, or has max_size lines but none of them is valid
};

// Reads <root>/tt-metal-cache.conf; nullopt when there is no such file. The last valid max_size line wins.
std::optional<ConfigValue> read_config(const fs::path& root) {
    const fs::path path = root / kKernelCacheConfigName;
    std::error_code ec;
    if (!fs::exists(fs::symlink_status(path, ec))) {
        return std::nullopt;
    }
    ConfigValue config;
    std::ifstream in(path);
    if (!in) {
        log_warning(tt::LogBuildKernels, "JIT kernel cache: cannot read {}: {}", path.string(), std::strerror(errno));
        config.invalid = true;
        return config;
    }
    const auto trim = [](std::string_view s) {
        const auto first = s.find_first_not_of(" \t\r");
        if (first == std::string_view::npos) {
            return std::string_view{};
        }
        return s.substr(first, s.find_last_not_of(" \t\r") - first + 1);
    };
    bool seen_invalid = false;
    std::string line;
    for (int line_no = 1; std::getline(in, line); line_no++) {
        const std::string_view text = trim(std::string_view(line).substr(0, line.find('#')));
        if (text.empty()) {
            continue;
        }
        const auto eq = text.find('=');
        const std::string_view key = trim(text.substr(0, eq));
        if (eq == std::string_view::npos || key != "max_size") {
            log_warning(tt::LogBuildKernels, "{}:{}: ignoring unknown setting '{}'", path.string(), line_no, text);
            continue;
        }
        const std::string_view value = trim(text.substr(eq + 1));
        const auto size = parse_cap(value, fmt::format("{}:{}: max_size", path.string(), line_no));
        if (!size) {
            seen_invalid = true;
            continue;
        }
        config.bytes = *size;
        config.text = value;
    }
    config.invalid = seen_invalid && config.text.empty();
    return config;
}

// Creates <root>/tt-metal-cache.conf holding max_size, unless the file already exists. Readers see either no
// file or the whole file: it is written under a temporary name and then renamed without replacing, so when
// several processes seed at once the first one wins and the others read its value.
void seed_config(const fs::path& root, std::string_view max_size) {
    static std::atomic<std::uint64_t> counter{0};
    const fs::path path = root / kKernelCacheConfigName;
    const fs::path tmp = root / fmt::format(".{}.{}-{}", kKernelCacheConfigName, ::getpid(), counter++);
    std::error_code ec;
    fs::create_directories(root, ec);
    {
        std::ofstream out(tmp);
        out << "# Written from TT_METAL_CACHE_MAX_SIZE. Every process using this cache reads this file;\n"
               "# it wins over the variable. Sizes as in ccache: 20G, 1.5G, 500M, 1Gi; a bare number is GiB;\n"
               "# 0 means no cap.\n"
            << "max_size = " << max_size << "\n";
        out.close();
        if (!out) {
            log_warning(tt::LogBuildKernels, "JIT kernel cache: cannot write {}", tmp.string());
            fs::remove(tmp, ec);
            return;
        }
    }
    int rc = ::renameat2(AT_FDCWD, tmp.c_str(), AT_FDCWD, path.c_str(), RENAME_NOREPLACE);
    if (rc != 0 && (errno == EINVAL || errno == ENOSYS)) {
        rc = ::link(tmp.c_str(), path.c_str());  // file system without RENAME_NOREPLACE; link never replaces either
    }
    const int err = errno;
    fs::remove(tmp, ec);
    if (rc == 0) {
        log_info(
            tt::LogBuildKernels,
            "JIT kernel cache: wrote max_size = {} from TT_METAL_CACHE_MAX_SIZE to {}; it stays and every process "
            "using this cache reads it. Edit or remove the file to change the cap.",
            max_size,
            path.string());
    } else if (err != EEXIST) {
        log_warning(tt::LogBuildKernels, "JIT kernel cache: cannot create {}: {}", path.string(), std::strerror(err));
    }
}

}  // namespace

std::uint64_t read_kernel_cache_max_size(const fs::path& root) {
    return read_config(root).value_or(ConfigValue{}).bytes;
}

std::uint64_t resolve_kernel_cache_max_size(const fs::path& root, const std::optional<std::string>& env_value) {
    std::optional<std::uint64_t> env_bytes;
    if (env_value) {
        env_bytes = parse_cap(*env_value, "TT_METAL_CACHE_MAX_SIZE");
    }
    auto config = read_config(root);
    if (!config && env_bytes) {
        seed_config(root, *env_value);
        config = read_config(root);  // another process may have seeded it first with its own value
        if (!config) {
            return *env_bytes;  // could not write the file (warned); still honour the variable here
        }
    }
    if (!config) {
        return 0;
    }
    const fs::path path = root / kKernelCacheConfigName;
    if (config->invalid) {
        if (env_bytes) {
            log_warning(
                tt::LogBuildKernels,
                "JIT kernel cache: {} has no valid max_size; using TT_METAL_CACHE_MAX_SIZE={}",
                path.string(),
                *env_value);
            return *env_bytes;
        }
        return 0;
    }
    if (env_bytes && *env_bytes != config->bytes) {
        log_warning(
            tt::LogBuildKernels,
            "TT_METAL_CACHE_MAX_SIZE={} differs from max_size = {} in {}; using the file, which every process "
            "sharing this cache reads. Edit or remove the file to change the cap.",
            *env_value,
            config->text.empty() ? "(not set)" : config->text,
            path.string());
    }
    return config->bytes;
}

void set_kernel_cache_cap(const fs::path& root, std::uint64_t max_bytes, std::chrono::seconds min_evict_age) {
    auto& upkeep = Upkeep::inst();
    std::lock_guard lock(upkeep.mutex);
    upkeep.root = root;
    upkeep.max_bytes = max_bytes;
    upkeep.min_evict_age = min_evict_age;
    upkeep.enabled = max_bytes != 0;
    if (max_bytes == 0) {
        upkeep.last_stamp.clear();
    }
}

void note_kernel_cache_dir_used(const fs::path& dir, std::chrono::seconds restamp_after) {
    auto& stamps = Upkeep::inst();
    if (!stamps.enabled) {
        return;
    }
    const std::string path = normalize(dir);
    const std::time_t now = std::time(nullptr);
    {
        std::lock_guard lock(stamps.mutex);
        auto [it, inserted] = stamps.last_stamp.try_emplace(path, now);
        if (!inserted) {
            if (now - it->second < static_cast<std::time_t>(restamp_after.count())) {
                return;
            }
            it->second = now;
        }
    }
    // Best effort: a stamp that fails only makes this entry look older than it is.
    ::utimensat(AT_FDCWD, path.c_str(), nullptr, 0);
}

namespace {
std::atomic<KernelCacheRaceHook> race_hook{nullptr};
}  // namespace

void set_kernel_cache_race_hook(KernelCacheRaceHook hook) { race_hook.store(hook); }

void kernel_cache_race_point(std::string_view point, const fs::path& dir) {
    if (auto* hook = race_hook.load()) {
        hook(point, dir);
    }
}

void create_kernel_cache_dir(const fs::path& dir) {
    const fs::path leaf = dir.has_filename() ? dir : dir.parent_path();  // the JIT passes "<variant>/"
    fs::create_directories(leaf.parent_path());
    kernel_cache_race_point("variant dir", leaf);
    std::error_code ec;
    fs::create_directory(leaf, ec);
    if (ec) {
        fs::create_directories(leaf);  // the parent was trimmed away meanwhile; a real error throws here
    }
}

KernelCacheTrimResult trim_kernel_cache(
    const fs::path& root, std::uint64_t max_bytes, std::chrono::seconds min_evict_age) {
    KernelCacheTrimResult result;
    if (max_bytes == 0 || !is_root_dir(root)) {
        return result;
    }
    const std::time_t now = std::time(nullptr);
    const std::time_t cutoff = now - static_cast<std::time_t>(min_evict_age.count());
    const auto start = std::chrono::steady_clock::now();
    TrimLock lock(root);
    if (!lock.held()) {
        return result;  // another process is trimming this cache
    }

    // Anything left in the trash belongs to a trimmer that died mid-delete or could not delete it; only lock
    // holders write it. What cannot be deleted stays, and the scan counts it.
    const fs::path trash = root / kTrashName;
    std::error_code ec;
    fs::remove_all(trash, ec);
    if (ec) {
        log_warning(tt::LogBuildKernels, "JIT kernel cache: cannot delete {}: {}", trash.string(), ec.message());
    }

    auto [variants, other] = scan(root);
    result.scanned = true;
    std::uint64_t total = other;
    for (const auto& v : variants) {
        total += v.size;
    }
    result.size_before = total;
    result.size_after = total;
    // The scan's real total replaces the counter, minus what this trim frees below.
    const auto save_counter = [&] { write_stats(root, {result.size_after, now, result.size_after}); };
    if (total <= max_bytes) {
        save_counter();
        return result;
    }

    const auto low_water = static_cast<std::uint64_t>(static_cast<double>(max_bytes) * kKernelCacheLowWaterFraction);
    if (other > low_water) {
        log_warning(
            tt::LogBuildKernels,
            "JIT kernel cache {}: cap {} is below the non-evictable {} (firmware and other files that are never "
            "evicted); not evicting. Raise max_size in {}.",
            root.string(),
            human_size(max_bytes),
            human_size(other),
            (root / kKernelCacheConfigName).string());
        save_counter();
        return result;
    }
    std::sort(variants.begin(), variants.end(), [](const Variant& a, const Variant& b) { return a.mtime < b.mtime; });
    fs::create_directories(trash, ec);

    static std::atomic<std::uint64_t> trash_counter{0};
    std::size_t undeleted = 0;
    std::string first_error;
    for (const auto& v : variants) {
        if (total <= low_water || v.mtime > cutoff) {
            break;  // sorted oldest first: everything after this is newer
        }
        if (mtime_of(v.path) > cutoff) {
            continue;  // stamped by another process since the scan
        }
        const fs::path staged = trash / fmt::format("{}-{}", ::getpid(), trash_counter++);
        if (::rename(v.path.c_str(), staged.c_str()) != 0) {
            const int err = errno;
            if (err != ENOENT) {
                log_debug(
                    tt::LogBuildKernels, "JIT kernel cache: cannot evict {}: {}", v.path.string(), std::strerror(err));
            }
            continue;  // ENOENT: someone else removed it already
        }
        result.variants_evicted++;
        fs::remove_all(staged, ec);
        std::uint64_t freed = v.size;
        if (ec) {
            SeenInodes seen;
            freed -= std::min(freed, disk_usage(staged, seen));
            if (undeleted++ == 0) {
                first_error = fmt::format("{}: {}", v.path.string(), ec.message());
            }
        }
        total -= freed;
        if (v.path.parent_path().filename() != kPchName) {
            ::rmdir(v.path.parent_path().c_str());  // the kernel's directory, once its last variant is gone
        }
    }
    fs::remove_all(trash, ec);
    if (undeleted > 0) {
        log_warning(
            tt::LogBuildKernels,
            "JIT kernel cache: could not delete {} evicted entries (first: {}); they stay in {} and count toward "
            "the cap",
            undeleted,
            first_error,
            trash.string());
    }

    result.trimmed = result.variants_evicted > 0;
    result.size_after = total;
    save_counter();
    if (result.trimmed) {
        const auto ms =
            std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::steady_clock::now() - start).count();
        log_info(
            tt::LogBuildKernels,
            "JIT kernel cache {} over its {} cap: evicted {} entries ({}), {} -> {} in {} ms",
            root.string(),
            human_size(max_bytes),
            result.variants_evicted,
            human_size(result.size_before - result.size_after),
            human_size(result.size_before),
            human_size(result.size_after),
            ms);
    }
    return result;
}

KernelCacheTrimResult maintain_kernel_cache(
    const fs::path& root, std::uint64_t max_bytes, std::chrono::seconds min_evict_age) {
    if (max_bytes == 0 || !is_root_dir(root)) {
        return {};
    }
    const auto stats = read_stats(root);
    if (stats && !scan_due(*stats, max_bytes, min_evict_age, std::time(nullptr))) {
        return {};  // the common case: one small file read, no walk of the cache
    }
    return trim_kernel_cache(root, max_bytes, min_evict_age);
}

std::optional<std::uint64_t> read_kernel_cache_size_counter(const fs::path& root) {
    const auto stats = read_stats(root);
    return stats ? std::optional(stats->size) : std::nullopt;
}

void track_kernel_cache_growth(const fs::path& dir, const std::function<void()>& build) {
    auto& upkeep = Upkeep::inst();
    if (!upkeep.enabled) {
        build();
        return;
    }
    SeenInodes seen_before;
    const std::uint64_t before = disk_usage(dir, seen_before);
    build();
    SeenInodes seen_after;
    const std::uint64_t after = disk_usage(dir, seen_after);
    if (after == before) {
        return;  // a cache hit: nothing was written
    }

    fs::path root;
    std::uint64_t max_bytes = 0;
    std::chrono::seconds min_evict_age{0};
    {
        std::lock_guard lock(upkeep.mutex);
        root = upkeep.root;
        max_bytes = upkeep.max_bytes;
        min_evict_age = upkeep.min_evict_age;
    }
    if (max_bytes == 0) {
        return;
    }
    bool trim = false;
    {
        TrimLock lock(root, /*wait=*/true);
        if (!lock.held()) {
            return;  // cannot open the lock file (read-only cache): nothing to keep up to date
        }
        auto stats = read_stats(root);
        if (stats) {
            stats->size =
                after > before ? stats->size + (after - before) : stats->size - std::min(stats->size, before - after);
            write_stats(root, *stats);
            trim = scan_due(*stats, max_bytes, min_evict_age, std::time(nullptr));
        } else {
            trim = true;  // missing or corrupt counter: rebuild it from a scan
        }
    }
    if (trim) {
        trim_kernel_cache(root, max_bytes, min_evict_age);
    }
}

}  // namespace tt::tt_metal
