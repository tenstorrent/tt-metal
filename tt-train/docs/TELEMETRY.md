# Telemetry Monitoring and Plotting

This guide shows how to correlate training performance with device telemetry (temperature, voltage, power, etc.).

## Quick Start

### 1. Run Training with Telemetry Monitoring

```bash
# Start telemetry polling (runs in background)
./scripts/poll_telemetry.sh telemetry.log

# Run your training
./build/sources/examples/nano_gpt/nano_gpt \
  -c ./configs/training_configs/your_config.yaml \
  > training.log 2>&1

# Stop telemetry polling
./scripts/poll_telemetry.sh stop
```

### 2. Generate Plots

```bash
python3 scripts/plot_telemetry.py \
  --training-log training.log \
  --telemetry-log telemetry.log \
  --output telemetry_plots.png
```

## What You Get

The plotting script generates a single PNG with multiple subplots:

**Training Metrics (always shown):**
- Loss vs Step
- Step Time vs Step

**Device Telemetry (when available):**
- Voltage vs Step
- Current vs Step
- Power vs Step
- AI Clock vs Step
- ASIC Temperature vs Step
- Heartbeat vs Step

The script automatically:
- Matches training steps to telemetry by timestamp
- Filters out compilation overhead (first 2 steps from step time)
- Works even if telemetry data is missing (shows training metrics only)
- Prints statistics for all metrics

## Polling Configuration

The telemetry polling interval is set in `scripts/poll_telemetry.sh`:

```bash
POLLING_INTERVAL_MS=1000  # Poll every 1 second
```

Adjust this based on your step time. Recommended: 2-3x more frequent than average step time.

## Troubleshooting

**No timestamps in training log:**
- Rebuild the training binary after pulling the timestamp changes
- Verify `main.cpp` includes the timestamp logging code

**Telemetry polling doesn't stop:**
```bash
cat /tmp/poll_telemetry.pid
kill <PID>
rm /tmp/poll_telemetry.pid
```

**No telemetry matches found:**
- Check both logs have data from the same time period
- Ensure polling started before training began
- Verify timestamps are present in training log

## Example Output

```
Parsing logs...
============================================================
Parsed 5000 training steps from training.log
  - 4998 steps have step time data
Parsed 10247 telemetry snapshots from telemetry.log

Matching logs by timestamp...
============================================================
Average step time: 0.206 seconds
Matched 4998 out of 5000 training steps
Discarded 2 unmatched steps

Generating plots...
============================================================
Plot saved to: telemetry_plots.png

Training Statistics:
------------------------------------------------------------
Loss                      - Min:   0.5234, Max:   4.6875, Mean:   1.2345
Step Time (ms)            - Min: 103.48, Max: 206.45, Mean: 205.32

Telemetry Statistics:
------------------------------------------------------------
Voltage (V)               - Min:     0.91, Max:     0.94, Mean:     0.92
Current (A)               - Min:    26.50, Max:    29.00, Mean:    27.43
Power (W)                 - Min:    24.10, Max:    27.30, Mean:    25.30
AI Clock (MHz)            - Min:  1000.00, Max:  1000.00, Mean:  1000.00
ASIC Temperature (°C)     - Min:    45.50, Max:    47.10, Mean:    46.11
Heartbeat                 - Min:  1559.00, Max:  1580.00, Mean:  1570.57
```


Example plots:
![](https://media.githubusercontent.com/media/tenstorrent/tutorial-assets/main/media/ttml/telemetry_plots_example.png)
