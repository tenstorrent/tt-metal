// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

// tt-smi with UMD telemetry: Enhanced version using TT-UMD APIs
// This version reads telemetry directly from device firmware via UMD
// instead of relying on sysfs

#include <iostream>
#include <iomanip>
#include <vector>
#include <map>
#include <set>
#include <string>
#include <sstream>
#include <fstream>
#include <filesystem>
#include <chrono>
#include <thread>
#include <memory>
#include <optional>
#include <sys/socket.h>
#include <sys/un.h>
#include <unistd.h>
#include <cstring>
#include <algorithm>
#include <dirent.h>
#include <limits.h>
#include <signal.h>
#include <sys/ioctl.h>
#include <deque>
#include <sstream>
#include <termios.h>  // Still needed for TerminalController (fallback)
#include <fcntl.h>    // Still needed for TerminalController (fallback)
#include <ncurses.h>  // ncurses TUI library (like nvtop uses)

// TT-UMD includes for direct device access
#include "umd/device/tt_device/tt_device.hpp"
#include "umd/device/firmware/firmware_info_provider.hpp"
#include "umd/device/types/arch.hpp"

// TT-Metal includes (for device enumeration fallback)
#include <tt-metalium/host_api.hpp>

#define TT_ALLOC_SERVER_SOCKET "/tmp/tt_allocation_server.sock"

namespace fs = std::filesystem;

// Message protocol (must match allocation server)
struct __attribute__((packed)) AllocMessage {
    enum Type : uint8_t {
        ALLOC = 1,
        FREE = 2,
        QUERY = 3,
        RESPONSE = 4,
        DUMP_REMAINING = 5,
        DEVICE_INFO_QUERY = 6,
        DEVICE_INFO_RESPONSE = 7,
        CB_ALLOC = 8,
        CB_FREE = 9,
        KERNEL_LOAD = 10,
        KERNEL_UNLOAD = 11,
        DUMP_KERNELS = 12
    };

    Type type;
    uint8_t pad1[3];
    int32_t device_id;
    uint64_t size;
    uint8_t buffer_type;
    uint8_t pad2[3];
    int32_t process_id;
    uint64_t buffer_id;
    uint64_t timestamp;

    uint64_t dram_allocated;
    uint64_t l1_allocated;
    uint64_t l1_small_allocated;
    uint64_t trace_allocated;
    uint64_t cb_allocated;      // NEW: Circular buffers
    uint64_t kernel_allocated;  // NEW: Kernel code

    uint64_t total_dram_size;
    uint64_t total_l1_size;
    uint32_t arch_type;
    uint32_t num_dram_channels;
    uint32_t dram_size_per_channel;
    uint32_t l1_size_per_core;
    uint32_t is_available;
    uint32_t num_devices;

    // Ring buffer stats (for real-time kernel L1 usage)
    uint32_t ringbuffer_total_size;    // Total ring buffer capacity (~67KB)
    uint32_t ringbuffer_used_bytes;    // Currently used by cached kernels
    uint32_t ringbuffer_num_programs;  // Number of programs cached
    uint32_t pad3;                     // Padding for alignment
};

struct ProcessInfo {
    pid_t pid;
    std::string name;
    int device_id;
    uint64_t dram_used;
    uint64_t l1_used;
    bool connected_to_server;
};

struct TelemetryData {
    double asic_temperature = -1.0;
    std::optional<double> board_temperature;
    std::optional<uint32_t> aiclk;      // MHz
    std::optional<uint32_t> axiclk;     // MHz
    std::optional<uint32_t> arcclk;     // MHz
    std::optional<uint32_t> fan_speed;  // RPM
    std::optional<uint32_t> tdp;        // Watts
    std::optional<uint32_t> tdc;        // Amps
    std::optional<uint32_t> vcore;      // mV
};

struct DeviceInfo {
    int device_id;
    std::string arch_name;
    uint64_t total_dram;
    uint64_t total_l1;
    uint64_t total_l1_small;
    uint64_t total_trace;
    uint64_t used_dram;
    uint64_t used_l1;
    uint64_t used_l1_small;
    uint64_t used_trace;
    uint64_t used_cb;      // NEW: Circular buffers
    uint64_t used_kernel;  // NEW: Kernel code

    // Ring buffer stats (real-time kernel L1 usage)
    uint32_t ringbuffer_total;
    uint32_t ringbuffer_used;
    uint32_t ringbuffer_programs;

    TelemetryData telemetry;
    std::vector<ProcessInfo> processes;
};

// ANSI colors
namespace Color {
const char* RESET = "\033[0m";
const char* BOLD = "\033[1m";
const char* GREEN = "\033[32m";
const char* YELLOW = "\033[33m";
const char* RED = "\033[31m";
const char* CYAN = "\033[36m";
const char* WHITE = "\033[37m";
const char* BLUE = "\033[34m";
const char* MAGENTA = "\033[35m";
}  // namespace Color

// History buffer for tracking metrics over time
template <typename T>
class MetricHistory {
private:
    std::deque<T> values_;
    size_t max_size_;

public:
    MetricHistory(size_t window_size = 60) : max_size_(window_size) {}

    void push(T value) {
        values_.push_back(value);
        while (values_.size() > max_size_) {
            values_.pop_front();
        }
    }

    const std::deque<T>& get_values() const { return values_; }
    size_t size() const { return values_.size(); }
    bool empty() const { return values_.empty(); }
    T latest() const { return values_.empty() ? T{} : values_.back(); }
};

// Per-device history
struct DeviceHistory {
    MetricHistory<double> dram_usage_pct;
    MetricHistory<uint64_t> dram_usage_bytes;
    MetricHistory<double> l1_usage_pct;
    MetricHistory<uint64_t> l1_usage_bytes;
    MetricHistory<uint64_t> l1_small_bytes;
    MetricHistory<uint64_t> trace_bytes;
    MetricHistory<double> temperature;
    MetricHistory<uint32_t> aiclk;

    DeviceHistory(size_t window_size = 60) :
        dram_usage_pct(window_size),
        dram_usage_bytes(window_size),
        l1_usage_pct(window_size),
        l1_usage_bytes(window_size),
        l1_small_bytes(window_size),
        trace_bytes(window_size),
        temperature(window_size),
        aiclk(window_size) {}
};

// ASCII Chart renderer
class ASCIIChart {
public:
    // Render a multi-line graph with smooth diagonal connections
    static std::vector<std::string> render_graph(
        const std::deque<double>& values, int width, int height, int graph_id = 0) {
        std::vector<std::string> lines(height, std::string(width, ' '));

        if (values.empty()) {
            return lines;
        }

        // Sample values to fit width
        std::vector<int> y_positions;
        for (int x = 0; x < width; x++) {
            size_t idx = (x * values.size()) / width;
            if (idx >= values.size()) {
                idx = values.size() - 1;
            }
            double pct = std::max(0.0, std::min(100.0, values[idx]));
            int y_pos = height - 1 - static_cast<int>((pct / 100.0) * (height - 1));
            y_pos = std::max(0, std::min(height - 1, y_pos));
            y_positions.push_back(y_pos);
        }

        // Draw with smooth transitions
        for (int x = 0; x < width && x < y_positions.size(); x++) {
            int y = y_positions[x];

            if (x > 0) {
                int prev_y = y_positions[x - 1];
                int diff = y - prev_y;

                if (diff == 0) {
                    // Flat horizontal
                    lines[y][x] = '-';
                } else if (diff == 1) {
                    // Single step down (descending)
                    lines[prev_y][x] = '\\';
                } else if (diff == -1) {
                    // Single step up (ascending)
                    lines[prev_y][x] = '/';
                } else if (diff > 1) {
                    // Multi-step down
                    lines[prev_y][x] = '\\';
                    for (int yy = prev_y + 1; yy < y; yy++) {
                        lines[yy][x] = '|';
                    }
                    lines[y][x] = '_';
                } else {
                    // Multi-step up
                    lines[prev_y][x] = '/';
                    for (int yy = y + 1; yy < prev_y; yy++) {
                        lines[yy][x] = '|';
                    }
                    lines[y][x] = '_';
                }
            } else {
                // First point
                lines[y][x] = '-';
            }
        }

        // Fill in horizontal connections
        for (int h = 0; h < height; h++) {
            for (int w = 1; w < width; w++) {
                if (lines[h][w] == ' ' && lines[h][w - 1] != ' ' && lines[h][w - 1] != '|') {
                    // Continue horizontal line
                    bool found_next = false;
                    for (int ww = w; ww < width && ww < w + 3; ww++) {
                        if (lines[h][ww] != ' ') {
                            found_next = true;
                            break;
                        }
                    }
                    if (found_next) {
                        lines[h][w] = '-';
                    }
                }
            }
        }

        return lines;
    }

    // Overlay two graphs (for DRAM and L1) with different characters
    static std::vector<std::string> render_dual_graph(
        const std::deque<double>& values1, const std::deque<double>& values2, int width, int height) {
        auto graph1 = render_graph(values1, width, height, 0);
        auto graph2 = render_graph(values2, width, height, 1);

        // Map DRAM to one set, L1 to another
        std::vector<std::string> combined(height, std::string(width, ' '));

        for (int h = 0; h < height; h++) {
            for (int w = 0; w < width; w++) {
                char c1 = graph1[h][w];
                char c2 = graph2[h][w];

                // Simple: L1 takes priority if present, otherwise DRAM
                if (c2 != ' ') {
                    // L1 line - use : for vertical to distinguish from DRAM
                    if (c2 == '|') {
                        combined[h][w] = ':';  // L1 vertical
                    } else {
                        combined[h][w] = c2;  // Use same characters as DRAM
                    }
                } else if (c1 != ' ') {
                    // DRAM line
                    combined[h][w] = c1;
                }
            }
        }

        return combined;
    }

    static std::vector<std::string> render_line_chart(
        const std::deque<double>& values, double min_val, double max_val, int width, int height) {
        std::vector<std::string> lines(height);

        if (values.empty()) {
            for (int h = 0; h < height; h++) {
                lines[h] = std::string(width, ' ');
            }
            return lines;
        }

        // Create 2D buffer
        std::vector<std::vector<char>> buffer(height, std::vector<char>(width, ' '));

        double range = max_val - min_val;
        if (range < 0.01) {
            range = 1.0;
        }

        // Plot points
        for (size_t i = 0; i < values.size() && i < width; i++) {
            double normalized = (values[i] - min_val) / range;
            int y = height - 1 - static_cast<int>(normalized * (height - 1));
            y = std::max(0, std::min(height - 1, y));

            size_t x = (i * width) / std::max(size_t(1), values.size());
            if (x < width) {
                buffer[y][x] = '#';
            }
        }

        // Convert buffer to strings
        for (int h = 0; h < height; h++) {
            lines[h] = std::string(buffer[h].begin(), buffer[h].end());
        }

        return lines;
    }
};

// Terminal control helper class
// Terminal controller for raw mode input (fallback when ncurses disabled)
class TerminalController {
private:
    struct termios orig_termios_;
    bool raw_mode_enabled_;

public:
    TerminalController() : raw_mode_enabled_(false) {}

    ~TerminalController() { disable_raw_mode(); }

    void enable_raw_mode() {
        if (raw_mode_enabled_) {
            return;
        }

        tcgetattr(STDIN_FILENO, &orig_termios_);
        struct termios raw = orig_termios_;
        raw.c_lflag &= ~(ICANON | ECHO);
        raw.c_cc[VMIN] = 0;
        raw.c_cc[VTIME] = 0;
        tcsetattr(STDIN_FILENO, TCSAFLUSH, &raw);

        int flags = fcntl(STDIN_FILENO, F_GETFL, 0);
        fcntl(STDIN_FILENO, F_SETFL, flags | O_NONBLOCK);

        raw_mode_enabled_ = true;
    }

    void disable_raw_mode() {
        if (!raw_mode_enabled_) {
            return;
        }
        tcsetattr(STDIN_FILENO, TCSAFLUSH, &orig_termios_);
        raw_mode_enabled_ = false;
    }

    char check_keypress() {
        char c = 0;
        read(STDIN_FILENO, &c, 1);
        return c;
    }
};

// NCurses wrapper for flicker-free TUI rendering (like nvtop/htop)
// Will be enabled once all std::cout calls are converted
class NCursesRenderer {
private:
    bool initialized_;
    int color_green_;
    int color_cyan_;
    int color_yellow_;
    int color_red_;
    int color_blue_;
    int color_white_;

public:
    NCursesRenderer() : initialized_(false) {}

    ~NCursesRenderer() { cleanup(); }

    bool init() {
        if (initialized_) {
            return true;
        }

        // Initialize ncurses
        initscr();              // Start ncurses mode
        cbreak();               // Disable line buffering
        noecho();               // Don't echo input
        nodelay(stdscr, TRUE);  // Non-blocking getch()
        keypad(stdscr, TRUE);   // Enable function keys
        curs_set(0);            // Hide cursor

        // Setup colors if supported
        if (has_colors()) {
            start_color();
            init_pair(1, COLOR_GREEN, COLOR_BLACK);
            init_pair(2, COLOR_CYAN, COLOR_BLACK);
            init_pair(3, COLOR_YELLOW, COLOR_BLACK);
            init_pair(4, COLOR_RED, COLOR_BLACK);
            init_pair(5, COLOR_BLUE, COLOR_BLACK);
            init_pair(6, COLOR_WHITE, COLOR_BLACK);

            color_green_ = COLOR_PAIR(1);
            color_cyan_ = COLOR_PAIR(2);
            color_yellow_ = COLOR_PAIR(3);
            color_red_ = COLOR_PAIR(4);
            color_blue_ = COLOR_PAIR(5);
            color_white_ = COLOR_PAIR(6);
        }

        initialized_ = true;
        return true;
    }

    void cleanup() {
        if (initialized_) {
            curs_set(1);  // Show cursor
            endwin();     // End ncurses mode
            initialized_ = false;
        }
    }

    void clear_screen() { clear(); }

    void refresh_screen() {
        refresh();  // This is the magic - double-buffered flush!
    }

    int get_key() {
        return getch();  // Non-blocking
    }

    void print_at(int row, int col, const std::string& text, int color_pair = 0) {
        if (color_pair > 0) {
            attron(COLOR_PAIR(color_pair));
        }
        mvprintw(row, col, "%s", text.c_str());
        if (color_pair > 0) {
            attroff(COLOR_PAIR(color_pair));
        }
    }

    void print_bold_at(int row, int col, const std::string& text) {
        attron(A_BOLD);
        mvprintw(row, col, "%s", text.c_str());
        attroff(A_BOLD);
    }

    // Print at current cursor position
    void print(const std::string& text) { printw("%s", text.c_str()); }

    // Print with color at current position
    void print_colored(const std::string& text, int color_pair) {
        if (color_pair > 0) {
            attron(COLOR_PAIR(color_pair));
        }
        printw("%s", text.c_str());
        if (color_pair > 0) {
            attroff(COLOR_PAIR(color_pair));
        }
    }

    // Move cursor to row, col
    void move_cursor(int row, int col) { move(row, col); }

    // Get current cursor position
    std::pair<int, int> get_cursor_position() {
        int y, x;
        getyx(stdscr, y, x);
        return {y, x};
    }

    int get_green_color() const { return 1; }
    int get_cyan_color() const { return 2; }
    int get_yellow_color() const { return 3; }
    int get_red_color() const { return 4; }
    int get_blue_color() const { return 5; }
    int get_white_color() const { return 6; }

    std::pair<int, int> get_screen_size() {
        int height, width;
        getmaxyx(stdscr, height, width);
        return {height, width};
    }
};

// Output wrapper that works with both cout and ncurses
class OutputWrapper {
private:
    NCursesRenderer* renderer_;
    bool use_ncurses_;

public:
    OutputWrapper(NCursesRenderer* renderer = nullptr) : renderer_(renderer), use_ncurses_(renderer != nullptr) {}

    // Stream-like output
    OutputWrapper& operator<<(const std::string& text) {
        if (use_ncurses_ && renderer_) {
            renderer_->print(text);
        } else {
            std::cout << text;
        }
        return *this;
    }

    OutputWrapper& operator<<(const char* text) { return *this << std::string(text); }

    OutputWrapper& operator<<(int val) { return *this << std::to_string(val); }

    OutputWrapper& operator<<(uint32_t val) { return *this << std::to_string(val); }

    OutputWrapper& operator<<(uint64_t val) { return *this << std::to_string(val); }

    OutputWrapper& operator<<(double val) { return *this << std::to_string(val); }

    // Handle std::endl and other manipulators
    OutputWrapper& operator<<(std::ostream& (*manip)(std::ostream&)) {
        if (use_ncurses_ && renderer_) {
            renderer_->print("\n");
        } else {
            std::cout << manip;
        }
        return *this;
    }

    // Flush (no-op for ncurses since it has its own refresh)
    void flush() {
        if (!use_ncurses_) {
            std::cout << std::flush;
        }
    }
};

// Global flag for terminal resize
static volatile sig_atomic_t g_resize_flag = 0;
static void handle_sigwinch(int sig) {
    (void)sig;
    g_resize_flag = 1;
    // Tell ncurses to handle resize
    if (stdscr != nullptr) {
        endwin();
        refresh();
    }
}

class TTSmiUMD {
private:
    int socket_fd_;
    bool server_available_;

    // Cache of UMD devices for telemetry (device_id -> TTDevice)
    mutable std::map<int, std::unique_ptr<tt::umd::TTDevice>> umd_device_cache_;

    // View mode: 1 = main, 2 = charts, 3 = detailed telemetry
    int current_view_;

    // NCurses renderer for flicker-free TUI
    mutable std::unique_ptr<NCursesRenderer> ncurses_renderer_;

    // Helper to clear to end of line (prevents artifacts from previous output)
    void clear_eol() const { std::cout << "\033[K"; }

    // History tracking for charts
    std::map<int, DeviceHistory> device_histories_;

    // Helper to read sysfs file as string
    static std::string read_sysfs_string(const std::string& path) {
        std::ifstream file(path);
        if (!file.is_open()) {
            return "";
        }
        std::string value;
        std::getline(file, value);
        return value;
    }

    // Helper to read sysfs file as integer
    static int64_t read_sysfs_int(const std::string& path) {
        std::string value = read_sysfs_string(path);
        if (value.empty()) {
            return -1;
        }
        try {
            return std::stoll(value);
        } catch (...) {
            return -1;
        }
    }

    // Get or create cached UMD device for telemetry
    tt::umd::TTDevice* get_umd_device(int device_id, bool show_status = false) const {
        // Check if device is already cached
        auto it = umd_device_cache_.find(device_id);
        if (it != umd_device_cache_.end()) {
            return it->second.get();
        }

        // Create and cache new device
        try {
            if (show_status) {
                std::cout << Color::YELLOW << "⚙️  Initializing device " << device_id << " for telemetry..."
                          << Color::RESET << std::flush;
            }

            std::unique_ptr<tt::umd::TTDevice> tt_device = tt::umd::TTDevice::create(device_id);
            if (!tt_device) {
                if (show_status) {
                    std::cout << " " << Color::RED << "Failed" << Color::RESET << std::endl;
                }
                return nullptr;
            }

            // Initialize the device (lightweight init for telemetry)
            tt_device->init_tt_device();

            if (show_status) {
                std::cout << " " << Color::GREEN << "Done" << Color::RESET << std::endl;
            }

            // Store in cache and return pointer
            auto* device_ptr = tt_device.get();
            umd_device_cache_[device_id] = std::move(tt_device);
            return device_ptr;

        } catch (const std::exception& e) {
            if (show_status) {
                std::cout << " " << Color::RED << "Error: " << e.what() << Color::RESET << std::endl;
            }
            return nullptr;
        }
    }

    // Read telemetry from a cached UMD device
    TelemetryData read_telemetry_from_cached_device(tt::umd::TTDevice* tt_device) const {
        TelemetryData data;

        try {
            if (!tt_device) {
                return data;
            }

            // Get firmware info provider for telemetry
            auto firmware_info = tt_device->get_firmware_info_provider();
            if (!firmware_info) {
                return data;
            }

            // Read telemetry from firmware
            double temp = firmware_info->get_asic_temperature();
            // Validate temperature is reasonable (max 100°C for Tenstorrent chips)
            if (temp >= -50.0 && temp <= 100.0) {
                data.asic_temperature = temp;
            }

            auto board_temp = firmware_info->get_board_temperature();
            if (board_temp.has_value() && board_temp.value() >= -50.0 && board_temp.value() <= 100.0) {
                data.board_temperature = board_temp;
            }

            auto aiclk = firmware_info->get_aiclk();
            if (aiclk.has_value() && aiclk.value() > 0 && aiclk.value() <= 1500) {  // Max 1500 MHz for TT chips
                data.aiclk = aiclk;
            }

            auto axiclk = firmware_info->get_axiclk();
            if (axiclk.has_value() && axiclk.value() > 0 && axiclk.value() <= 1500) {
                data.axiclk = axiclk;
            }

            auto arcclk = firmware_info->get_arcclk();
            if (arcclk.has_value() && arcclk.value() > 0 && arcclk.value() <= 1500) {
                data.arcclk = arcclk;
            }

            auto fan_speed = firmware_info->get_fan_speed();
            if (fan_speed.has_value() && fan_speed.value() < 20000) {  // Max reasonable RPM
                data.fan_speed = fan_speed;
            }

            auto tdp = firmware_info->get_tdp();
            if (tdp.has_value() && tdp.value() > 0 && tdp.value() <= 300) {  // Max 300W for TT chips
                data.tdp = tdp;
            }

            auto tdc = firmware_info->get_tdc();
            if (tdc.has_value() && tdc.value() > 0 && tdc.value() <= 350) {  // Max 350A for TT chips
                data.tdc = tdc;
            }

            auto vcore = firmware_info->get_vcore();
            if (vcore.has_value() && vcore.value() > 0 && vcore.value() <= 900) {  // Max 0.9V (900mV) for TT chips
                data.vcore = vcore;
            }

        } catch (const std::exception& e) {
            // Silently fail - device might not be accessible
            // Fall back to empty telemetry
        }

        return data;
    }

    // Read telemetry from device by ID (wrapper for convenience)
    TelemetryData read_telemetry_from_device(int device_id) {
        auto* tt_device = get_umd_device(device_id, false);
        return read_telemetry_from_cached_device(tt_device);
    }

    // Alias for backward compatibility
    TelemetryData read_telemetry_sysfs(int device_id) { return read_telemetry_from_device(device_id); }

    // Get process name from PID
    std::string get_process_name(pid_t pid) {
        std::string comm_path = "/proc/" + std::to_string(pid) + "/comm";
        std::ifstream comm_file(comm_path);
        if (comm_file) {
            std::string name;
            std::getline(comm_file, name);
            return name;
        }
        return "unknown";
    }

    // Get command line for a process
    std::string get_process_cmdline(pid_t pid) {
        std::string cmdline_path = "/proc/" + std::to_string(pid) + "/cmdline";
        std::ifstream cmdline_file(cmdline_path);
        if (cmdline_file) {
            std::string cmdline;
            std::getline(cmdline_file, cmdline, '\0');
            // Extract just the executable name
            size_t last_slash = cmdline.find_last_of('/');
            if (last_slash != std::string::npos) {
                return cmdline.substr(last_slash + 1);
            }
            return cmdline.empty() ? get_process_name(pid) : cmdline;
        }
        return get_process_name(pid);
    }

    // Discover processes using Tenstorrent devices by scanning /proc
    std::set<pid_t> discover_processes_using_devices() {
        std::set<pid_t> pids;

        try {
            // Walk through /proc/[pid]/fd/ to find processes with tenstorrent devices open
            for (const auto& entry : fs::directory_iterator("/proc")) {
                if (!entry.is_directory()) {
                    continue;
                }

                std::string dirname = entry.path().filename().string();
                if (!std::all_of(dirname.begin(), dirname.end(), ::isdigit)) {
                    continue;
                }

                pid_t pid = std::stoi(dirname);

                // Check /proc/[pid]/fd/ for tenstorrent device files
                std::string fd_path = entry.path().string() + "/fd";
                try {
                    for (const auto& fd_entry : fs::directory_iterator(fd_path)) {
                        if (!fs::is_symlink(fd_entry)) {
                            continue;
                        }

                        fs::path target = fs::read_symlink(fd_entry);
                        std::string target_str = target.string();

                        if (target_str.find("/dev/tenstorrent/") != std::string::npos) {
                            pids.insert(pid);
                            break;
                        }
                    }
                } catch (...) {
                    // Permission denied or process exited
                    continue;
                }
            }
        } catch (const std::exception& e) {
            std::cerr << "Warning: Error scanning /proc: " << e.what() << std::endl;
        }

        return pids;
    }

    // Try to connect to allocation server
    bool try_connect_to_server() {
        socket_fd_ = socket(AF_UNIX, SOCK_STREAM, 0);
        if (socket_fd_ < 0) {
            return false;
        }

        struct sockaddr_un addr;
        memset(&addr, 0, sizeof(addr));
        addr.sun_family = AF_UNIX;
        strncpy(addr.sun_path, TT_ALLOC_SERVER_SOCKET, sizeof(addr.sun_path) - 1);

        if (connect(socket_fd_, (struct sockaddr*)&addr, sizeof(addr)) < 0) {
            close(socket_fd_);
            socket_fd_ = -1;
            return false;
        }

        return true;
    }

    // Query device info from server
    AllocMessage query_device_info(int device_id) {
        AllocMessage query;
        memset(&query, 0, sizeof(query));
        query.type = AllocMessage::DEVICE_INFO_QUERY;
        query.device_id = device_id;

        send(socket_fd_, &query, sizeof(query), 0);

        AllocMessage response;
        recv(socket_fd_, &response, sizeof(response), 0);
        return response;
    }

    // Query allocation stats from server
    AllocMessage query_device_stats(int device_id) {
        AllocMessage query;
        memset(&query, 0, sizeof(query));
        query.type = AllocMessage::QUERY;
        query.device_id = device_id;

        send(socket_fd_, &query, sizeof(query), 0);

        AllocMessage response;
        recv(socket_fd_, &response, sizeof(response), 0);
        return response;
    }

    // Get number of devices
    uint32_t get_num_devices() {
        AllocMessage query;
        memset(&query, 0, sizeof(query));
        query.type = AllocMessage::DEVICE_INFO_QUERY;
        query.device_id = -1;  // Special: query for count

        send(socket_fd_, &query, sizeof(query), 0);

        AllocMessage response;
        recv(socket_fd_, &response, sizeof(response), 0);
        return response.num_devices;
    }

    // Format bytes
    std::string format_bytes(uint64_t bytes) {
        const char* units[] = {"B", "KB", "MB", "GB", "TB"};
        int unit = 0;
        double size = static_cast<double>(bytes);

        while (size >= 1024.0 && unit < 4) {
            size /= 1024.0;
            unit++;
        }

        std::ostringstream oss;
        oss << std::fixed << std::setprecision(1) << size << units[unit];
        return oss.str();
    }

    // Get architecture name
    const char* get_arch_name(uint32_t arch_type) {
        switch (arch_type) {
            case 1: return "Grayskull";
            case 2: return "Wormhole_B0";
            case 3: return "Blackhole";
            case 4: return "Quasar";
            default: return "Unknown";
        }
    }

    // Print progress bar
    std::string get_bar(double percentage, int width = 20) {
        std::ostringstream bar;
        bar << "[";
        int filled = static_cast<int>((percentage / 100.0) * width);

        const char* color = Color::GREEN;
        if (percentage >= 90.0) {
            color = Color::RED;
        } else if (percentage >= 75.0) {
            color = Color::YELLOW;
        }

        for (int i = 0; i < width; ++i) {
            if (i < filled) {
                bar << color << "█" << Color::RESET;
            } else {
                bar << "░";
            }
        }
        bar << "]";
        return bar.str();
    }

    // Print device table header
    void print_device_header() {
        std::cout << Color::BOLD << Color::CYAN;
        std::cout
            << "┌────────────────────────────────────────────────────────────────────────────────────────────────────┐"
            << std::endl;
        std::cout << "│ ";

        // Get current time
        auto now = std::chrono::system_clock::now();
        auto time_t = std::chrono::system_clock::to_time_t(now);
        auto tm = *std::localtime(&time_t);

        std::cout << std::left << std::setw(40) << "tt-smi-umd v1.0 (with UMD telemetry)";
        std::cout << std::right << std::setw(39);

        std::ostringstream time_str;
        time_str << std::put_time(&tm, "%a %b %d %H:%M:%S %Y");
        std::cout << time_str.str();
        std::cout << " │" << std::endl;

        std::cout
            << "├────────────────────────────────────────────────────────────────────────────────────────────────────┤"
            << std::endl;
        std::cout << "│ " << std::left << std::setw(4) << "GPU" << std::setw(16) << "Name" << std::setw(10) << "Temp"
                  << std::setw(10) << "Power" << std::setw(12) << "AICLK" << std::setw(48) << "Memory-Usage"
                  << " │" << std::endl;
        std::cout
            << "├────────────────────────────────────────────────────────────────────────────────────────────────────┤"
            << Color::RESET << std::endl;
    }

    void print_device_footer() {
        std::cout << Color::BOLD << Color::CYAN;
        std::cout
            << "└────────────────────────────────────────────────────────────────────────────────────────────────────┘";
        std::cout << Color::RESET << std::endl;
    }

    void print_process_header() {
        std::cout << "\n" << Color::BOLD << Color::CYAN;
        std::cout << "Processes:" << Color::RESET << std::endl;
        std::cout << Color::BOLD << Color::CYAN;
        std::cout
            << "┌────────────────────────────────────────────────────────────────────────────────────────────────────┐"
            << std::endl;
        std::cout << "│ " << std::left << std::setw(8) << "PID" << std::setw(20) << "Name" << std::setw(8) << "Device"
                  << std::setw(12) << "DRAM" << std::setw(12) << "L1" << std::setw(36) << "Status"
                  << " │" << std::endl;
        std::cout
            << "├────────────────────────────────────────────────────────────────────────────────────────────────────┤"
            << Color::RESET << std::endl;
    }

    void print_process_footer() {
        std::cout << Color::BOLD << Color::CYAN;
        std::cout
            << "└────────────────────────────────────────────────────────────────────────────────────────────────────┘";
        std::cout << Color::RESET << std::endl;
    }

public:
    TTSmiUMD() : socket_fd_(-1), server_available_(false), current_view_(1) {}

    ~TTSmiUMD() {
        if (socket_fd_ >= 0) {
            close(socket_fd_);
        }
        // device_cache_ will be auto-cleaned by unique_ptr
    }

    void run(bool watch_mode = false, int refresh_ms = 1000, bool use_sysfs = false) {
        static bool first_run = true;
        TerminalController term_ctrl;  // For keyboard input when ncurses disabled

        // TODO: NCurses support temporarily disabled until all std::cout calls are converted
        // Initialize ncurses for watch mode (flicker-free TUI like nvtop)
        bool use_ncurses = false;  // Set to true once all rendering is converted
        if (watch_mode && !ncurses_renderer_ && use_ncurses) {
            ncurses_renderer_ = std::make_unique<NCursesRenderer>();
            if (!ncurses_renderer_->init()) {
                std::cerr << "Failed to initialize ncurses" << std::endl;
                return;
            }

            // Setup signal handler for terminal resize
            struct sigaction sa;
            memset(&sa, 0, sizeof(sa));
            sa.sa_handler = handle_sigwinch;
            sa.sa_flags = SA_RESTART;
            sigaction(SIGWINCH, &sa, NULL);
        }

        // Fall back to old ANSI approach for now
        if (watch_mode && first_run && !use_ncurses) {
            // Use alternate screen buffer (like nvtop, htop, vim)
            std::cout << "\033[?1049h";  // Switch to alternate buffer
            std::cout << "\033[?25l";    // Hide cursor
            std::cout << std::flush;
            term_ctrl.enable_raw_mode();  // Enable keyboard input
        }

        do {
            // Handle terminal resize
            if (g_resize_flag) {
                g_resize_flag = 0;
                if (ncurses_renderer_) {
                    // ncurses handles resize automatically via endwin/refresh in signal handler
                }
            }

            // Check for keyboard input
            if (watch_mode) {
                int key = 0;
                if (ncurses_renderer_) {
                    // NCurses mode
                    key = ncurses_renderer_->get_key();
                } else {
                    // Fallback mode
                    key = term_ctrl.check_keypress();
                }

                if (key == '1') {
                    current_view_ = 1;  // Main view
                } else if (key == '2') {
                    current_view_ = 2;  // Charts view
                } else if (key == '3') {
                    current_view_ = 3;  // Detailed telemetry view
                } else if (key == 'q' || key == 'Q') {
                    break;  // Exit
                }
            }

            // Clear screen and prepare for rendering
            if (watch_mode && ncurses_renderer_) {
                // NCurses mode
                ncurses_renderer_->clear_screen();
            } else if (watch_mode) {
                // Fallback ANSI mode
                std::cout << "\033[2J";  // Clear screen
                std::cout << "\033[H";   // Move to top
            }

            // Create output wrapper for this iteration (will use std::cout for now)
            OutputWrapper out(nullptr);  // Disable ncurses routing until all cout converted

            // Show initialization status on first run
            if (first_run && !use_sysfs && !watch_mode) {
                out << Color::CYAN << "Initializing UMD telemetry..." << Color::RESET << std::endl;
            }

            // Try to connect to allocation server
            server_available_ = try_connect_to_server();

            if (!server_available_ && !watch_mode) {
                out << Color::YELLOW << "⚠ Allocation server not running" << Color::RESET << std::endl;
                out << "  Start it with: ./allocation_server_poc" << std::endl;
                out << "\n  Showing telemetry only (no memory tracking)\n" << std::endl;
            }

            // Discover processes
            auto pids = discover_processes_using_devices();

            // Get device info
            std::vector<DeviceInfo> devices;
            uint32_t num_devices = 0;

            if (server_available_) {
                num_devices = get_num_devices();

                for (uint32_t i = 0; i < num_devices; ++i) {
                    auto dev_info = query_device_info(i);
                    auto stats = query_device_stats(i);

                    DeviceInfo dev;
                    dev.device_id = i;
                    dev.arch_name = get_arch_name(dev_info.arch_type);
                    dev.total_dram = dev_info.total_dram_size;
                    dev.total_l1 = dev_info.total_l1_size;
                    dev.total_l1_small = dev_info.total_l1_size;  // L1_SMALL is part of L1 space
                    dev.total_trace = dev_info.total_l1_size;     // TRACE is also part of L1 space
                    dev.used_dram = stats.dram_allocated;
                    dev.used_l1 = stats.l1_allocated;
                    dev.used_l1_small = stats.l1_small_allocated;
                    dev.used_trace = stats.trace_allocated;
                    dev.used_cb = stats.cb_allocated;          // NEW
                    dev.used_kernel = stats.kernel_allocated;  // NEW

                    // Ring buffer stats (not available without opening device)
                    dev.ringbuffer_total = 0;
                    dev.ringbuffer_used = 0;
                    dev.ringbuffer_programs = 0;

                    // Read telemetry via UMD or sysfs (show status on first run)
                    if (use_sysfs) {
                        dev.telemetry = read_telemetry_sysfs(i);
                    } else {
                        auto* tt_device = get_umd_device(i, first_run);
                        if (tt_device) {
                            dev.telemetry = read_telemetry_from_cached_device(tt_device);
                        } else {
                            // Device initialization failed - likely in use by another process
                            dev.telemetry = TelemetryData();  // Empty telemetry
                        }
                    }

                    devices.push_back(dev);
                }
            } else {
                // Without server, enumerate devices manually
                // This is a simplified approach - in production, use proper device discovery
                for (int i = 0; i < 8; ++i) {  // Try up to 8 devices
                    DeviceInfo dev;
                    dev.device_id = i;
                    dev.arch_name = "Unknown";
                    dev.total_dram = 0;
                    dev.total_l1 = 0;
                    dev.total_l1_small = 0;
                    dev.total_trace = 0;
                    dev.used_dram = 0;
                    dev.used_l1 = 0;
                    dev.used_l1_small = 0;
                    dev.used_trace = 0;
                    dev.used_cb = 0;
                    dev.used_kernel = 0;

                    // Try to read telemetry (show status on first run)
                    if (use_sysfs) {
                        dev.telemetry = read_telemetry_sysfs(i);
                    } else {
                        auto* tt_device = get_umd_device(i, first_run);
                        if (tt_device) {
                            dev.telemetry = read_telemetry_from_cached_device(tt_device);
                        } else {
                            // Device initialization failed - likely in use by another process
                            dev.telemetry = TelemetryData();  // Empty telemetry
                        }
                    }

                    // Only add if we got valid telemetry
                    if (dev.telemetry.asic_temperature >= 0) {
                        devices.push_back(dev);
                    }
                }
                num_devices = devices.size();
            }

            if (first_run && !use_sysfs) {
                // Check if any devices failed to initialize
                bool any_failed = (num_devices == 0 || umd_device_cache_.size() < num_devices);
                if (any_failed) {
                    std::cout << Color::YELLOW
                              << "⚠  Some devices couldn't be initialized (may be in use by another process)"
                              << Color::RESET << std::endl;
                    std::cout << Color::CYAN << "   Tip: Close other tools using the devices (like tt-smi -r)"
                              << Color::RESET << std::endl
                              << std::endl;
                } else {
                    std::cout << Color::GREEN << "✓ Telemetry initialized" << Color::RESET << std::endl << std::endl;
                }
                first_run = false;
            }

            // Populate process info
            for (pid_t pid : pids) {
                ProcessInfo proc;
                proc.pid = pid;
                proc.name = get_process_cmdline(pid);
                proc.device_id = 0;  // TODO: Detect which device via fdinfo
                proc.dram_used = 0;
                proc.l1_used = 0;
                proc.connected_to_server = false;

                // Associate with device 0 for now
                if (!devices.empty()) {
                    devices[0].processes.push_back(proc);
                }
            }

            // Print device table
            print_device_header();

            for (const auto& dev : devices) {
                std::cout << Color::BOLD << "│ " << Color::RESET;
                std::cout << std::left << std::setw(4) << dev.device_id;
                std::cout << std::setw(16) << dev.arch_name;

                // Temperature (from UMD telemetry)
                if (dev.telemetry.asic_temperature >= 0) {
                    std::cout << std::fixed << std::setprecision(1) << std::setw(10)
                              << (std::to_string(static_cast<int>(dev.telemetry.asic_temperature)) + "°C");
                } else {
                    std::cout << std::setw(10) << "N/A";
                }

                // Power (from UMD telemetry - use TDP)
                if (dev.telemetry.tdp.has_value()) {
                    std::cout << std::setw(10) << (std::to_string(dev.telemetry.tdp.value()) + "W");
                } else {
                    std::cout << std::setw(10) << "N/A";
                }

                // AICLK (from UMD telemetry)
                if (dev.telemetry.aiclk.has_value()) {
                    std::cout << std::setw(12) << (std::to_string(dev.telemetry.aiclk.value()) + " MHz");
                } else {
                    std::cout << std::setw(12) << "N/A";
                }

                // Memory usage (no bar, just text like nvidia-smi)
                if (server_available_ && dev.total_dram > 0) {
                    std::ostringstream mem_str;
                    mem_str << format_bytes(dev.used_dram) << "/" << format_bytes(dev.total_dram);
                    std::cout << std::setw(48) << mem_str.str();
                } else {
                    std::cout << std::setw(48) << "N/A";
                }

                std::cout << " │" << std::endl;
            }

            print_device_footer();

            // Update history for all devices
            for (const auto& dev : devices) {
                if (device_histories_.find(dev.device_id) == device_histories_.end()) {
                    device_histories_[dev.device_id] = DeviceHistory(60);
                }

                auto& hist = device_histories_[dev.device_id];

                // Update memory usage
                if (dev.total_dram > 0) {
                    double dram_pct = (static_cast<double>(dev.used_dram) / dev.total_dram) * 100.0;
                    hist.dram_usage_pct.push(dram_pct);
                    hist.dram_usage_bytes.push(dev.used_dram);
                }

                if (dev.total_l1 > 0) {
                    double l1_pct = (static_cast<double>(dev.used_l1) / dev.total_l1) * 100.0;
                    hist.l1_usage_pct.push(l1_pct);
                    hist.l1_usage_bytes.push(dev.used_l1);
                }

                // Track L1_SMALL and TRACE
                hist.l1_small_bytes.push(dev.used_l1_small);
                hist.trace_bytes.push(dev.used_trace);

                // Update telemetry
                if (dev.telemetry.asic_temperature >= 0) {
                    hist.temperature.push(dev.telemetry.asic_temperature);
                }
                if (dev.telemetry.aiclk.has_value()) {
                    hist.aiclk.push(dev.telemetry.aiclk.value());
                }
            }

            // View 1: Main View - Show memory breakdown (nvidia-smi style)
            if (current_view_ == 1) {
                // Print memory breakdown (if server available)
                if (server_available_ && !devices.empty()) {
                    std::cout << "\n" << Color::BOLD << Color::CYAN << "Memory Breakdown:" << Color::RESET << std::endl;
                    for (const auto& dev : devices) {
                        if (dev.total_dram == 0) {
                            continue;  // Skip devices without info
                        }

                        std::cout << "\n"
                                  << Color::BOLD << "Device " << dev.device_id << " (" << dev.arch_name
                                  << "):" << Color::RESET << std::endl;
                        std::cout << std::string(70, '-') << std::endl;

                        // DRAM with bar chart and percentage
                        double dram_util =
                            (dev.total_dram > 0) ? (static_cast<double>(dev.used_dram) / dev.total_dram) * 100.0 : 0.0;
                        std::cout << "  DRAM:       " << std::setw(10) << std::left << format_bytes(dev.used_dram)
                                  << std::right << " / " << std::setw(10) << std::left << format_bytes(dev.total_dram)
                                  << std::right << "  " << get_bar(dram_util, 25) << " " << std::fixed
                                  << std::setprecision(1) << dram_util << "%" << std::endl;

                        // L1 Memory with bar chart and percentage (total of all L1 components)
                        uint64_t total_l1_used =
                            dev.used_l1 + dev.used_cb + dev.ringbuffer_used;  // Use ringbuffer_used for kernels
                        double l1_util =
                            (dev.total_l1 > 0) ? (static_cast<double>(total_l1_used) / dev.total_l1) * 100.0 : 0.0;
                        std::cout << "  L1 Memory:  " << std::setw(10) << std::left << format_bytes(total_l1_used)
                                  << std::right << " / " << std::setw(10) << std::left << format_bytes(dev.total_l1)
                                  << std::right << "  " << get_bar(l1_util, 25) << " " << std::fixed
                                  << std::setprecision(1) << l1_util << "%" << std::endl;

                        // L1 sub-components (text only, no bars)
                        if (dev.used_l1 > 0) {
                            std::cout << "    Buffers:  " << std::setw(12) << format_bytes(dev.used_l1) << std::endl;
                        }
                        if (dev.used_cb > 0) {
                            std::cout << "    CBs:      " << std::setw(12) << format_bytes(dev.used_cb) << std::endl;
                        }
                        // Show tracked kernel allocations (Application, Fabric, Dispatch)
                        if (dev.used_kernel > 0) {
                            std::cout << "    Kernels:  " << std::setw(12) << format_bytes(dev.used_kernel)
                                      << std::endl;
                        }

                        // Query for additional memory types (L1_SMALL, TRACE)
                        auto stats = query_device_stats(dev.device_id);

                        // L1_SMALL (text only)
                        if (stats.l1_small_allocated > 0) {
                            std::cout << "    L1_SMALL: " << std::setw(12) << format_bytes(stats.l1_small_allocated)
                                      << std::endl;
                        }

                        // TRACE (text only)
                        if (stats.trace_allocated > 0) {
                            std::cout << "    TRACE:    " << std::setw(12) << format_bytes(stats.trace_allocated)
                                      << std::endl;
                        }
                    }
                }

                // Print process table for View 1
                if (!pids.empty()) {
                    print_process_header();

                    for (const auto& dev : devices) {
                        for (const auto& proc : dev.processes) {
                            std::cout << "│ ";
                            std::cout << std::left << std::setw(8) << proc.pid;

                            // Truncate name if too long
                            std::string name = proc.name;
                            if (name.length() > 18) {
                                name = name.substr(0, 15) + "...";
                            }
                            std::cout << std::setw(20) << name;
                            std::cout << std::setw(8) << proc.device_id;

                            if (server_available_ && proc.connected_to_server) {
                                std::cout << std::setw(12) << format_bytes(proc.dram_used);
                                std::cout << std::setw(12) << format_bytes(proc.l1_used);
                                std::cout << std::setw(36)
                                          << (Color::GREEN + std::string("Connected to server") + Color::RESET);
                            } else {
                                std::cout << std::setw(12) << "N/A";
                                std::cout << std::setw(12) << "N/A";
                                std::cout << std::setw(36)
                                          << (Color::YELLOW + std::string("Device open (no tracking)") + Color::RESET);
                            }

                            std::cout << " │" << std::endl;
                        }
                    }

                    print_process_footer();
                } else {
                    std::cout << "\n"
                              << Color::YELLOW << "No processes using Tenstorrent devices" << Color::RESET << std::endl;
                }
            }

            // View 2: Charts View - Matrix layout for all devices
            if (current_view_ == 2 && !devices.empty()) {
                const int CHART_WIDTH = 60;   // Width per chart
                const int CHART_HEIGHT = 12;  // Height per chart

                // Calculate rectangular matrix layout for better screen usage
                int num_devices = devices.size();
                int cols = 2;  // Default to 2 columns
                int rows = 1;

                // Layouts: prefer 2 columns, then grow rows, then move to more columns
                if (num_devices == 1) {
                    cols = 1;
                    rows = 1;
                } else if (num_devices == 2) {
                    cols = 2;
                    rows = 1;  // 2x1
                } else if (num_devices == 3 || num_devices == 4) {
                    cols = 2;
                    rows = 2;  // 2x2
                } else if (num_devices <= 6) {
                    cols = 2;
                    rows = 3;  // 2x3 for 5-6 devices
                } else if (num_devices <= 8) {
                    cols = 2;
                    rows = 4;  // 2x4 for 7-8 devices
                } else if (num_devices <= 12) {
                    cols = 3;
                    rows = (num_devices + 2) / 3;  // 3x3 or 3x4
                } else {
                    cols = 4;
                    rows = (num_devices + 3) / 4;  // 4x4, 4x5, etc.
                }

                std::cout << Color::CYAN << "Chart View - Last 60 seconds (" << num_devices << " device"
                          << (num_devices > 1 ? "s" : "") << " in " << rows << "x" << cols << " matrix)" << Color::RESET
                          << std::endl;
                std::cout << Color::YELLOW << "(Press 1 for main view)" << Color::RESET << std::endl;
                std::cout << std::endl;

                int devices_to_show = num_devices;

                // Render all devices (vertical layout for simplicity, but show ALL devices)
                for (int d = 0; d < devices_to_show; d++) {
                    const auto& dev = devices[d];

                    // Skip if no history yet
                    if (device_histories_.find(dev.device_id) == device_histories_.end()) {
                        continue;
                    }

                    const auto& hist = device_histories_[dev.device_id];

                    // Device header
                    std::cout << Color::BOLD << "Device " << dev.device_id << " [" << dev.arch_name << "]"
                              << Color::RESET;
                    std::cout << " TEMP ";
                    if (dev.telemetry.asic_temperature >= 0) {
                        std::cout << Color::YELLOW << std::setw(4) << std::fixed << std::setprecision(0)
                                  << dev.telemetry.asic_temperature << "C" << Color::RESET;
                    } else {
                        std::cout << "  N/A ";
                    }
                    std::cout << " CLK ";
                    if (dev.telemetry.aiclk.has_value()) {
                        std::cout << Color::BLUE << std::setw(4) << dev.telemetry.aiclk.value() << "MHz"
                                  << Color::RESET;
                    } else {
                        std::cout << " N/A  ";
                    }
                    clear_eol();
                    std::cout << std::endl;

                    // Usage bars
                    std::cout << "DRAM[";
                    int dram_bar_len = 25;
                    int dram_filled = static_cast<int>((hist.dram_usage_pct.latest() / 100.0) * dram_bar_len);
                    std::cout << Color::GREEN << std::string(dram_filled, '|')
                              << std::string(dram_bar_len - dram_filled, ' ') << Color::RESET;
                    std::cout << std::setw(6) << std::fixed << std::setprecision(1) << hist.dram_usage_pct.latest()
                              << "%";
                    std::cout << std::setw(8) << format_bytes(hist.dram_usage_bytes.latest()) << "/" << std::setw(8)
                              << std::left << format_bytes(dev.total_dram) << std::right << "]";

                    std::cout << " L1[";
                    int l1_bar_len = 15;
                    int l1_filled = static_cast<int>((hist.l1_usage_pct.latest() / 100.0) * l1_bar_len);
                    std::cout << Color::CYAN << std::string(l1_filled, '|') << std::string(l1_bar_len - l1_filled, ' ')
                              << Color::RESET;
                    std::cout << std::setw(6) << std::fixed << std::setprecision(1) << hist.l1_usage_pct.latest()
                              << "%]";
                    clear_eol();
                    std::cout << std::endl;

                    // Fixed-size graph box
                    std::cout << "  +" << std::string(CHART_WIDTH, '-') << "+";
                    clear_eol();
                    std::cout << std::endl;

                    // Render dual graph
                    auto combined_graph = ASCIIChart::render_dual_graph(
                        hist.dram_usage_pct.get_values(),
                        hist.l1_usage_pct.get_values(),
                        CHART_WIDTH - 4,  // Leave room for borders and labels
                        CHART_HEIGHT);

                    for (int h = 0; h < CHART_HEIGHT; h++) {
                        // Y-axis label (fixed width)
                        int pct_val = 100 - (h * 100 / (CHART_HEIGHT - 1));
                        if (pct_val % 20 == 0) {
                            std::cout << std::setw(3) << pct_val;
                        } else {
                            std::cout << "   ";
                        }

                        std::cout << "|";

                        // Graph content (fixed width) with simple color coding
                        for (int w = 0; w < CHART_WIDTH - 4; w++) {
                            char c = combined_graph[h][w];
                            if (c == ':') {
                                // L1 vertical - cyan to distinguish
                                std::cout << Color::CYAN << c << Color::RESET;
                            } else if (c != ' ') {
                                // All other line characters - green
                                std::cout << Color::GREEN << c << Color::RESET;
                            } else {
                                std::cout << c;
                            }
                        }

                        std::cout << "|";

                        // Legend (showing what's actually graphed)
                        if (h == 2) {
                            std::cout << " " << Color::GREEN << "DRAM" << Color::RESET << " " << std::fixed
                                      << std::setprecision(1) << std::setw(5) << hist.dram_usage_pct.latest() << "%";
                        } else if (h == 4) {
                            std::cout << " " << Color::CYAN << "L1 (:)" << Color::RESET << " " << std::fixed
                                      << std::setprecision(1) << std::setw(5) << hist.l1_usage_pct.latest() << "%";
                        } else if (h == 6) {
                            // Show L1_SMALL as text only (not graphed)
                            double l1_small_pct = dev.total_l1_small > 0
                                                      ? (hist.l1_small_bytes.latest() * 100.0 / dev.total_l1_small)
                                                      : 0.0;
                            std::cout << " L1_SMALL: " << std::fixed << std::setprecision(1) << l1_small_pct << "%";
                        } else if (h == 8) {
                            // Show TRACE as text only (not graphed)
                            double trace_pct =
                                dev.total_trace > 0 ? (hist.trace_bytes.latest() * 100.0 / dev.total_trace) : 0.0;
                            std::cout << " TRACE: " << std::fixed << std::setprecision(1) << trace_pct << "%";
                        }

                        clear_eol();
                        std::cout << std::endl;
                    }

                    std::cout << "  +" << std::string(CHART_WIDTH, '-') << "+";
                    clear_eol();
                    std::cout << std::endl;

                    std::cout << "   " << std::setw(6) << "60s ago" << std::string(CHART_WIDTH - 18, ' ') << "now";
                    clear_eol();
                    std::cout << std::endl;

                    // Box separator between devices
                    std::cout << std::endl;
                    if (d < devices_to_show - 1) {  // Don't show separator after last device
                        std::cout << Color::CYAN << "  " << std::string(CHART_WIDTH + 4, '-') << Color::RESET
                                  << std::endl;
                        std::cout << std::endl;
                    }
                }

                // Fill remaining space to prevent artifacts
                for (int i = 0; i < 5; i++) {
                    clear_eol();
                    std::cout << std::endl;
                }
            }

            // View 3: Detailed Telemetry View
            if (current_view_ == 3 && !devices.empty()) {
                std::cout << "\n" << Color::BOLD << Color::CYAN << "Device Telemetry:" << Color::RESET << std::endl;
                for (const auto& dev : devices) {
                    std::cout << "\n"
                              << Color::BOLD << "Device " << dev.device_id << " (" << dev.arch_name
                              << "):" << Color::RESET << std::endl;
                    std::cout << std::string(70, '-') << std::endl;

                    // Temperature
                    std::cout << "  Temperature: ";
                    if (dev.telemetry.asic_temperature >= 0) {
                        std::cout << Color::GREEN << std::fixed << std::setprecision(1)
                                  << dev.telemetry.asic_temperature << "°C" << Color::RESET;
                    } else {
                        std::cout << Color::YELLOW << "N/A" << Color::RESET;
                    }

                    if (dev.telemetry.board_temperature.has_value()) {
                        std::cout << "  (Board: " << std::fixed << std::setprecision(1)
                                  << dev.telemetry.board_temperature.value() << "°C)";
                    }
                    std::cout << std::endl;

                    // Power & Current
                    std::cout << "  Power:       ";
                    if (dev.telemetry.tdp.has_value()) {
                        std::cout << Color::GREEN << dev.telemetry.tdp.value() << "W" << Color::RESET;
                    } else {
                        std::cout << Color::YELLOW << "N/A" << Color::RESET;
                    }

                    if (dev.telemetry.tdc.has_value()) {
                        std::cout << "  (Current: " << dev.telemetry.tdc.value() << "A)";
                    }
                    std::cout << std::endl;

                    // Voltage
                    std::cout << "  Voltage:     ";
                    if (dev.telemetry.vcore.has_value()) {
                        std::cout << Color::GREEN << dev.telemetry.vcore.value() << "mV" << Color::RESET << " ("
                                  << std::fixed << std::setprecision(3) << (dev.telemetry.vcore.value() / 1000.0)
                                  << "V)";
                    } else {
                        std::cout << Color::YELLOW << "N/A" << Color::RESET;
                    }
                    std::cout << std::endl;

                    // Clock Frequencies
                    std::cout << "  Clocks:" << std::endl;
                    std::cout << "    AICLK:     ";
                    if (dev.telemetry.aiclk.has_value()) {
                        std::cout << Color::GREEN << dev.telemetry.aiclk.value() << " MHz" << Color::RESET;
                    } else {
                        std::cout << Color::YELLOW << "N/A" << Color::RESET;
                    }
                    std::cout << std::endl;

                    std::cout << "    AXICLK:    ";
                    if (dev.telemetry.axiclk.has_value()) {
                        std::cout << Color::GREEN << dev.telemetry.axiclk.value() << " MHz" << Color::RESET;
                    } else {
                        std::cout << Color::YELLOW << "N/A" << Color::RESET;
                    }
                    std::cout << std::endl;

                    std::cout << "    ARCCLK:    ";
                    if (dev.telemetry.arcclk.has_value()) {
                        std::cout << Color::GREEN << dev.telemetry.arcclk.value() << " MHz" << Color::RESET;
                    } else {
                        std::cout << Color::YELLOW << "N/A" << Color::RESET;
                    }
                    std::cout << std::endl;

                    // Fan Speed
                    if (dev.telemetry.fan_speed.has_value()) {
                        std::cout << "  Fan Speed:   " << Color::GREEN << dev.telemetry.fan_speed.value() << " RPM"
                                  << Color::RESET << std::endl;
                    }
                }
            }

            // Print footer info
            std::cout << "\n"
                      << Color::CYAN << "💡 Telemetry source:" << Color::RESET
                      << (use_sysfs ? " sysfs" : " UMD (direct firmware access)") << std::endl;

            if (!server_available_) {
                std::cout << Color::CYAN << "💡 TIP:" << Color::RESET
                          << " For memory tracking, start the allocation server:" << std::endl;
                std::cout << "   ./allocation_server_poc" << std::endl;
            }

            if (watch_mode) {
                std::cout << "\n"
                          << Color::CYAN << "📋 View:" << Color::RESET << " Press " << Color::BOLD << "1"
                          << Color::RESET << " main, " << Color::BOLD << "2" << Color::RESET << " charts, "
                          << Color::BOLD << "3" << Color::RESET << " telemetry, " << Color::BOLD << "q" << Color::RESET
                          << " quit" << std::endl;
                std::string view_name;
                if (current_view_ == 1) {
                    view_name = "Main View";
                } else if (current_view_ == 2) {
                    view_name = "Charts View";
                } else if (current_view_ == 3) {
                    view_name = "Detailed Telemetry";
                }
                std::cout << Color::CYAN << "   Current:" << Color::RESET << " " << view_name;
                clear_eol();
                std::cout << std::endl;
            }

            if (watch_mode) {
                // Refresh screen
                if (ncurses_renderer_) {
                    ncurses_renderer_->refresh_screen();
                    napms(refresh_ms);
                } else {
                    // Fallback ANSI mode
                    std::cout << std::flush;
                    std::this_thread::sleep_for(std::chrono::milliseconds(refresh_ms));
                }
            }

            // Close connection for this iteration
            if (socket_fd_ >= 0) {
                close(socket_fd_);
                socket_fd_ = -1;
            }

        } while (watch_mode);

        // Cleanup on exit
        if (watch_mode) {
            if (ncurses_renderer_) {
                ncurses_renderer_->cleanup();
                ncurses_renderer_.reset();
            } else {
                // Fallback ANSI cleanup
                std::cout << "\033[?25h";    // Show cursor
                std::cout << "\033[?1049l";  // Return to main screen buffer
                std::cout << std::flush;
            }
        }
    }
};

int main(int argc, char* argv[]) {
    bool watch_mode = false;
    int refresh_ms = 500;  // Default 500ms like nvtop
    bool use_sysfs = false;

    // Parse arguments
    for (int i = 1; i < argc; ++i) {
        std::string arg = argv[i];
        if (arg == "-w" || arg == "--watch") {
            watch_mode = true;
        } else if (arg == "-r" && i + 1 < argc) {
            refresh_ms = std::stoi(argv[++i]);
        } else if (arg == "--sysfs") {
            use_sysfs = true;
        } else if (arg == "-h" || arg == "--help") {
            std::cout << "tt-smi-umd: Tenstorrent System Management Interface (with UMD telemetry)" << std::endl;
            std::cout << "\nUsage: " << argv[0] << " [OPTIONS]" << std::endl;
            std::cout << "\nOptions:" << std::endl;
            std::cout << "  -w, --watch    Watch mode (refresh continuously)" << std::endl;
            std::cout << "  -r <ms>        Refresh interval in milliseconds (default: 1000)" << std::endl;
            std::cout << "  --sysfs        Use sysfs instead of UMD for telemetry (comparison mode)" << std::endl;
            std::cout << "  -h, --help     Show this help" << std::endl;
            std::cout << "\nExamples:" << std::endl;
            std::cout << "  " << argv[0] << "              # Show current state (UMD telemetry)" << std::endl;
            std::cout << "  " << argv[0] << " -w            # Watch mode with UMD telemetry" << std::endl;
            std::cout << "  " << argv[0] << " --sysfs       # Use sysfs telemetry (limited)" << std::endl;
            std::cout << "  " << argv[0] << " -w -r 500     # Watch with 500ms refresh" << std::endl;
            std::cout << "\nFeatures:" << std::endl;
            std::cout << "  • Direct firmware telemetry via UMD (temperature, power, clocks)" << std::endl;
            std::cout << "  • Works with local AND remote devices" << std::endl;
            std::cout << "  • More detailed than sysfs-based monitoring" << std::endl;
            std::cout << "  • Memory tracking via allocation server (if running)" << std::endl;
            return 0;
        }
    }

    try {
        TTSmiUMD smi;
        smi.run(watch_mode, refresh_ms, use_sysfs);
    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << std::endl;
        return 1;
    }

    return 0;
}
