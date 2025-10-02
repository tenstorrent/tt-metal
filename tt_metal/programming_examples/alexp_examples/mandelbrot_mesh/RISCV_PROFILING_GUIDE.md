# TT-Metal RISC-V Profiling Guide

## Overview

This guide shows you how to enable comprehensive RISC-V profiling for TT-Metal applications, including the Mandelbrot mesh implementation.

## 🔧 Environment Variables for Profiling

### Basic Profiling Setup
```bash
# Enable profiler
export TT_METAL_PROFILER=1

# Disable DPRINT (conflicts with profiler)
unset TT_METAL_DPRINT_CORES

# Set profiler log level
export TT_METAL_PROFILER_LOG_LEVEL=INFO

# Enable device profiling
export TT_METAL_DEVICE_PROFILER=1
```

### Advanced Profiling Options
```bash
# Enable all RISC-V cores profiling
export TT_METAL_PROFILER_CORES="all"

# Or specify specific cores
export TT_METAL_PROFILER_CORES="0,0;1,1;2,2"

# Enable kernel profiling
export PROFILE_KERNEL=1

# Enable trace profiling
export TT_METAL_PROFILER_TRACE=1

# Set profiler buffer size (in MB)
export TT_METAL_PROFILER_BUFFER_SIZE=64

# Enable NOC event profiling
export TT_METAL_NOC_EVENT_PROFILER=1
```

### Profiler Output Options
```bash
# Set profiler output directory
export TT_METAL_PROFILER_OUTPUT_DIR="./profiler_logs"

# Enable CSV output
export TT_METAL_PROFILER_CSV=1

# Enable Tracy profiler integration
export TT_METAL_TRACY_ENABLE=1

# Set profiler file prefix
export TT_METAL_PROFILER_FILE_PREFIX="mandelbrot_mesh"
```

## 📊 Profiling Types Available

### 1. **Kernel Profiler**
Profiles individual kernel execution:
```bash
export PROFILE_KERNEL=1
export TT_METAL_KERNEL_PROFILER=1
```

### 2. **NOC Event Profiler**
Profiles Network-on-Chip events:
```bash
export TT_METAL_NOC_EVENT_PROFILER=1
export TT_METAL_NOC_PROFILER_CORES="all"
```

### 3. **Dispatch Core Profiler**
Profiles dispatch cores:
```bash
export TT_METAL_DISPATCH_PROFILER=1
export PROFILER_OPT_DO_DISPATCH_CORES=1
```

### 4. **Fabric Event Profiler**
Profiles fabric/mesh communication:
```bash
export TT_METAL_FABRIC_EVENT_PROFILER=1
```

### 5. **Custom Cycle Count Profiler**
For detailed cycle counting:
```bash
export TT_METAL_CUSTOM_CYCLE_PROFILER=1
```

## 🚀 Complete Profiling Setup Script

Create this script to enable all RISC-V profiling:

```bash
#!/bin/bash
# save as: enable_full_profiling.sh

echo "🔧 Enabling comprehensive TT-Metal RISC-V profiling..."

# Core profiler settings
export TT_METAL_PROFILER=1
export TT_METAL_DEVICE_PROFILER=1
export PROFILE_KERNEL=1

# Disable conflicting features
unset TT_METAL_DPRINT_CORES

# Enable all profiler types
export TT_METAL_KERNEL_PROFILER=1
export TT_METAL_NOC_EVENT_PROFILER=1
export TT_METAL_DISPATCH_PROFILER=1
export TT_METAL_FABRIC_EVENT_PROFILER=1
export TT_METAL_CUSTOM_CYCLE_PROFILER=1

# Profiler configuration
export TT_METAL_PROFILER_CORES="all"
export TT_METAL_PROFILER_TRACE=1
export TT_METAL_PROFILER_BUFFER_SIZE=128
export TT_METAL_PROFILER_LOG_LEVEL=DEBUG

# Output settings
export TT_METAL_PROFILER_OUTPUT_DIR="./profiler_output"
export TT_METAL_PROFILER_CSV=1
export TT_METAL_PROFILER_FILE_PREFIX="riscv_profile"

# Tracy integration (optional)
export TT_METAL_TRACY_ENABLE=1

# Create output directory
mkdir -p $TT_METAL_PROFILER_OUTPUT_DIR

echo "✅ Profiling enabled! Output will be in: $TT_METAL_PROFILER_OUTPUT_DIR"
echo ""
echo "📊 Enabled profiling types:"
echo "   • Kernel Profiler"
echo "   • NOC Event Profiler"
echo "   • Dispatch Core Profiler"
echo "   • Fabric Event Profiler"
echo "   • Custom Cycle Profiler"
echo ""
echo "🎯 Now run your TT-Metal application..."
```

## 🎯 Running Mandelbrot with Full Profiling

### Method 1: Using Environment Variables
```bash
# Enable profiling
source enable_full_profiling.sh

# Run C++ version with profiling
cd /home/tt-metal-apv/tt_metal/programming_examples/alexp_examples/mandelbrot_mesh
./build_and_run.sh

# Run Python version with profiling
python3 python_mandelbrot_mesh.py
```

### Method 2: Inline Profiling
```bash
TT_METAL_PROFILER=1 \
TT_METAL_DEVICE_PROFILER=1 \
PROFILE_KERNEL=1 \
TT_METAL_PROFILER_CORES="all" \
TT_METAL_PROFILER_OUTPUT_DIR="./mandelbrot_profiling" \
./build_and_run.sh
```

## 📈 Profiler Output Analysis

### Generated Files
After running with profiling enabled, you'll find:

```
profiler_output/
├── riscv_profile_device_0.csv          # Device-level profiling
├── riscv_profile_kernel_events.csv     # Kernel execution events
├── riscv_profile_noc_events.csv        # NOC traffic analysis
├── riscv_profile_dispatch.csv          # Dispatch core activity
├── riscv_profile_fabric.csv            # Mesh communication
└── riscv_profile_cycles.csv            # Cycle count details
```

### Key Metrics to Look For

1. **Kernel Execution Time**:
   ```
   Kernel Name, Core, Start Time, End Time, Duration (cycles)
   mandelbrot_compute, (0,0), 1000, 5000, 4000
   ```

2. **NOC Traffic**:
   ```
   Source Core, Dest Core, Bytes Transferred, Bandwidth
   (0,0), (1,1), 2048, 1.2 GB/s
   ```

3. **RISC-V Core Utilization**:
   ```
   Core Type, Core ID, Active Cycles, Idle Cycles, Utilization %
   BRISC, 0, 4500, 500, 90%
   NCRISC, 0, 4200, 800, 84%
   ```

### Viewing Results
```bash
# View kernel profiling
cat profiler_output/riscv_profile_kernel_events.csv | column -t -s','

# View NOC events
cat profiler_output/riscv_profile_noc_events.csv | head -20

# View dispatch profiling
cat profiler_output/riscv_profile_dispatch.csv
```

## 🔍 Profiling Analysis Tools

### Built-in Analysis Scripts
```bash
# Run profiler analysis
cd $TT_METAL_HOME
python -m tt_metal.tools.profiler.process_device_log profiler_output/

# Generate profiler report
python -m tt_metal.tools.profiler.generate_report profiler_output/
```

### Custom Analysis
```python
import pandas as pd
import matplotlib.pyplot as plt

# Load kernel profiling data
df = pd.read_csv('profiler_output/riscv_profile_kernel_events.csv')

# Plot kernel execution times
df.groupby('kernel_name')['duration_cycles'].mean().plot(kind='bar')
plt.title('Average Kernel Execution Time')
plt.ylabel('Cycles')
plt.show()
```

## 🐛 Troubleshooting

### Common Issues

1. **Profiler Not Starting**:
   ```bash
   # Check conflicts
   unset TT_METAL_DPRINT_CORES
   export TT_METAL_PROFILER=1
   ```

2. **Missing Output Files**:
   ```bash
   # Ensure output directory exists
   mkdir -p $TT_METAL_PROFILER_OUTPUT_DIR

   # Check permissions
   chmod 755 $TT_METAL_PROFILER_OUTPUT_DIR
   ```

3. **Buffer Overflow**:
   ```bash
   # Increase buffer size
   export TT_METAL_PROFILER_BUFFER_SIZE=256
   ```

### Verification Commands
```bash
# Check profiler status
echo "Profiler enabled: $TT_METAL_PROFILER"
echo "Output directory: $TT_METAL_PROFILER_OUTPUT_DIR"

# List profiler environment variables
env | grep TT_METAL_PROFILER | sort

# Test profiler with simple example
cd $TT_METAL_HOME/tt_metal/programming_examples/profiler/test_custom_cycle_count
make && ./test_custom_cycle_count
```

## 📊 Expected Profiling Results for Mandelbrot

When running the Mandelbrot mesh implementation with full profiling, you should see:

### Kernel Metrics
- **mandelbrot_compute**: High cycle count due to iterative computation
- **mandelbrot_writer**: Lower cycle count for data movement
- **Parallelization efficiency**: 8 devices working simultaneously

### NOC Traffic
- **Inter-device communication**: Mesh topology data movement
- **Memory bandwidth**: DRAM read/write patterns
- **Load balancing**: Even distribution across 2×4 mesh

### Performance Insights
- **Compute vs Memory bound**: Ratio analysis
- **Mesh utilization**: All 8 devices active
- **Bottleneck identification**: Slowest kernel/core

This comprehensive profiling will give you detailed insights into the RISC-V execution patterns and performance characteristics of your Mandelbrot mesh computation! 🚀📊
