# Chart View Design for tt_smi_umd

## Overview
Add a third interactive view (View 3) to `tt_smi_umd` that displays memory usage and telemetry over time using ASCII charts, similar to nvtop.

## View 3: Charts View (Press `3`)

### Layout
```
┌────────────────────────────────────────────────────────────────────────┐
│ tt-smi-umd v1.0 - Charts View                  Mon Nov 03 23:30:00 2025│
├────────────────────────────────────────────────────────────────────────┤
│                          Device 0: Blackhole                            │
├────────────────────────────────────────────────────────────────────────┤
│ DRAM Usage (Last 60s)                                     31.9GB Total │
│ 100%│                                                                   │
│  80%│                                                        ╭──╮       │
│  60%│                                    ╭───────────────────╯  ╰───   │
│  40%│          ╭─────────────────────────╯                             │
│  20%│  ╭───────╯                                                        │
│   0%└──┴────┴────┴────┴────┴────┴────┴────┴────┴────┴────┴────┴────   │
│      60s  50s  40s  30s  20s  10s   0s                                 │
│                                                                         │
│ L1 Usage (Last 60s)                                       306.0MB Total │
│ 100%│                                                                   │
│  80%│                                                                   │
│  60%│        ╭╮                        ╭╮                     ╭╮        │
│  40%│      ╭─╯╰╮    ╭──╮          ╭───╯╰───╮             ╭──╯╰─╮      │
│  20%│  ╭───╯   ╰────╯  ╰──────────╯        ╰─────────────╯     ╰──    │
│   0%└──┴────┴────┴────┴────┴────┴────┴────┴────┴────┴────┴────┴────   │
│      60s  50s  40s  30s  20s  10s   0s                                 │
│                                                                         │
│ Temperature (Last 60s)                                                 │
│ 100°│                                                                   │
│  75°│                                                                   │
│  50°│  ──────────────────────────────────────────────────────          │
│  25°│                                                                   │
│   0°└──┴────┴────┴────┴────┴────┴────┴────┴────┴────┴────┴────┴────   │
│      60s  50s  40s  30s  20s  10s   0s                                 │
│                                                                         │
│ Power (Last 60s)                                                       │
│ 300W│                                                                   │
│ 200W│                                                                   │
│ 100W│                                                                   │
│  50W│  ──────────────────────────────────────────────────────          │
│   0W└──┴────┴────┴────┴────┴────┴────┴────┴────┴────┴────┴────┴────   │
│      60s  50s  40s  30s  20s  10s   0s                                 │
└────────────────────────────────────────────────────────────────────────┘

📋 View: Press 1 for main, 2 for telemetry, 3 for charts, q to quit
   Current: Charts View | Showing Device 0 | Press ← → to switch devices
```

### Features

#### 1. **Memory Charts**
- **DRAM Usage**: Shows DRAM utilization percentage and absolute size over time
- **L1 Usage**: Shows L1 utilization percentage over time
- **L1_SMALL Usage**: If allocated, shows separate chart
- **TRACE Usage**: If allocated, shows separate chart

#### 2. **Telemetry Charts**
- **Temperature**: ASIC temperature over time (0-100°C range)
- **Power**: Power consumption over time (0-300W range)
- **Current**: Current draw over time (0-350A range)
- **Voltage**: Core voltage over time (0-900mV range)
- **Clock Frequencies**: AICLK, AXICLK, ARCCLK over time

#### 3. **Time Window**
- Default: Last 60 seconds
- Configurable: 30s, 60s, 120s, 300s (5min)
- Update frequency: Matches refresh rate (default 1000ms)

#### 4. **Multi-Device Support**
- Use `←` and `→` arrow keys to switch between devices
- Show one device at a time to avoid clutter
- Display device ID and name in header

### Implementation Details

#### Data Structures

```cpp
// History buffer for each metric
template<typename T>
class MetricHistory {
private:
    std::deque<T> values_;
    size_t max_size_;

public:
    MetricHistory(size_t window_size) : max_size_(window_size) {}

    void push(T value) {
        values_.push_back(value);
        if (values_.size() > max_size_) {
            values_.pop_front();
        }
    }

    const std::deque<T>& get_values() const { return values_; }
};

// Per-device history
struct DeviceHistory {
    MetricHistory<double> dram_usage_pct;    // Percentage
    MetricHistory<uint64_t> dram_usage_bytes; // Absolute bytes
    MetricHistory<double> l1_usage_pct;
    MetricHistory<double> temperature;
    MetricHistory<uint32_t> power;
    MetricHistory<uint32_t> current;
    MetricHistory<uint32_t> voltage;
    MetricHistory<uint32_t> aiclk;
    MetricHistory<uint32_t> axiclk;
    MetricHistory<uint32_t> arcclk;

    DeviceHistory(size_t window_size = 60) :
        dram_usage_pct(window_size),
        dram_usage_bytes(window_size),
        l1_usage_pct(window_size),
        temperature(window_size),
        power(window_size),
        current(window_size),
        voltage(window_size),
        aiclk(window_size),
        axiclk(window_size),
        arcclk(window_size) {}
};

// Global history tracker
std::map<int, DeviceHistory> device_histories;
```

#### ASCII Chart Rendering

```cpp
class ASCIIChart {
public:
    static std::string render_line_chart(
        const std::deque<double>& values,
        double min_val,
        double max_val,
        int width,
        int height
    ) {
        // Create 2D char buffer
        std::vector<std::vector<char>> buffer(height, std::vector<char>(width, ' '));

        // Draw chart lines, scale values to fit height
        // Use box-drawing characters: ─ │ ╭ ╮ ╰ ╯
        // ...

        // Convert buffer to string
        std::string result;
        for (const auto& row : buffer) {
            result += std::string(row.begin(), row.end()) + "\n";
        }
        return result;
    }
};
```

#### Interactive Controls

- `1`: Switch to Main View
- `2`: Switch to Detailed Telemetry View
- `3`: Switch to Charts View
- `←`: Previous device (in Charts View)
- `→`: Next device (in Charts View)
- `t`: Toggle time window (30s/60s/120s/300s)
- `q`: Quit

### Implementation Plan

1. **Phase 1**: Add data history tracking
   - Create `MetricHistory` and `DeviceHistory` classes
   - Update telemetry collection to push to history
   - Limit memory usage with circular buffers

2. **Phase 2**: Implement ASCII chart rendering
   - Create `ASCIIChart` helper class
   - Implement line chart renderer with box-drawing characters
   - Add axis labels and scaling

3. **Phase 3**: Add View 3 UI
   - Create chart layout
   - Render each metric chart
   - Add device switching with arrow keys

4. **Phase 4**: Polish and optimize
   - Add time window configuration
   - Optimize rendering performance
   - Add color coding for different metrics

### Benefits

- **Visual Trends**: See memory usage patterns over time
- **Performance Monitoring**: Identify spikes and anomalies
- **Resource Planning**: Understand typical usage patterns
- **Debugging**: Correlate memory usage with application behavior
- **Similar to nvtop**: Familiar interface for GPU monitoring

### Future Enhancements

- Export chart data to CSV
- Configurable chart types (line, bar, sparkline)
- Per-process memory usage charts
- Network bandwidth charts (if applicable)
- Multi-device comparison view
