# TT-Metal Kernel Debugging Guide

## 🔍 **Overview**

This guide shows you how to debug TT-Metal kernels and see what's happening on each RISC-V core during execution. The Mandelbrot kernels have been instrumented with debug prints to show you exactly what's happening.

## 🛠️ **Debug Setup**

### **Environment Variables for Kernel Debugging**

```bash
# Required: Set TT_METAL_HOME
export TT_METAL_HOME="/home/tt-metal-apv"

# Enable DPRINT (kernel debug prints)
export TT_METAL_DPRINT_CORES="all"           # Debug all cores
export TT_METAL_DPRINT_ENABLE=1              # Enable DPRINT system
export TT_METAL_DPRINT_DISABLE_ASSERT=1      # Disable assertions for cleaner output

# Debug output configuration
export TT_METAL_DPRINT_FILE="./kernel_debug.log"   # Output file
export TT_METAL_DPRINT_PRINT_ALL=1                 # Print all debug messages

# Application logging
export TT_METAL_LOG_LEVEL=Debug
export TT_METAL_LOGGER_LEVEL=Debug

# Better debugging experience
export TT_METAL_SLOW_DISPATCH_MODE=1         # Slower but more debuggable

# IMPORTANT: Disable profiler (conflicts with DPRINT)
unset TT_METAL_PROFILER
unset TT_METAL_DEVICE_PROFILER
```

### **⚠️ Key Constraints**

- **DPRINT and Profiler are mutually exclusive** - you can't use both at the same time
- **Hardware Required**: Real TT hardware or simulator mode for kernel execution
- **Core Specification**: Use `"all"` or specific cores like `"0,0;1,1;2,2"`

## 🚀 **Running Debug Scripts**

### **Method 1: Hardware Debug**
```bash
cd /home/tt-metal-apv/tt_metal/programming_examples/alexp_examples/mandelbrot_mesh
chmod +x debug_kernels.sh
./debug_kernels.sh
```

### **Method 2: CPU Simulation Debug**
```bash
chmod +x debug_kernels_cpu.sh
./debug_kernels_cpu.sh
```

### **Method 3: Manual Debug Setup**
```bash
# Set environment
export TT_METAL_HOME="/home/tt-metal-apv"
export TT_METAL_DPRINT_CORES="all"
export TT_METAL_DPRINT_ENABLE=1
export TT_METAL_DPRINT_FILE="./my_debug.log"

# Run application
/home/tt-metal-apv/build-cmake/programming_examples/mandelbrot_mesh

# View debug output
cat ./my_debug.log
```

## 📊 **Debug Output Analysis**

### **Expected Debug Messages**

#### **Compute Kernel Debug Output:**
```
MANDELBROT COMPUTE KERNEL STARTED
Device ID: 0
Num tiles: 16
Image size: 512x512
Max iterations: 100
Coordinate bounds: x[-2.5, 1.5] y[-2, 2]
Coordinate deltas: dx=0.0078125 dy=0.0078125
Processing tile 0/16
Pixel(0,0) c=(-2.5,-2) iter=2
Pixel(1,0) c=(-2.492188,-2) iter=3
Pixel(2,0) c=(-2.484375,-2) iter=4
...
Completed tile 0
Processing tile 1/16
...
```

#### **Dataflow Kernel Debug Output:**
```
MANDELBROT WRITER KERNEL STARTED
Dst addr: 0x10000000
Num tiles: 16
Tile size: 2048 bytes
Writing tile 0/16
CB addr: 0x20000000
Completed tile 0
Writing tile 1/16
...
MANDELBROT WRITER KERNEL COMPLETED - wrote 16 tiles
```

### **Debug Message Categories**

| **Pattern** | **Meaning** | **What to Look For** |
|-------------|-------------|---------------------|
| `MANDELBROT COMPUTE KERNEL STARTED` | Compute kernel launched | Should see 8 instances (one per mesh device) |
| `MANDELBROT WRITER KERNEL STARTED` | Dataflow kernel launched | Should see 8 instances (one per mesh device) |
| `Device ID: X` | Which mesh device is processing | Values 0-7 for 2×4 mesh |
| `Pixel(x,y) c=(cx,cy) iter=N` | Actual Mandelbrot calculation | Real complex coordinates and iteration counts |
| `Writing tile X/Y` | Data being written to DRAM | Tiles being stored successfully |
| `Coordinate bounds: x[a,b] y[c,d]` | Complex plane mapping | Each device gets different coordinate ranges |

## 🔧 **Debugging Different Scenarios**

### **Scenario 1: Kernels Not Starting**
**Symptoms:**
- No `KERNEL STARTED` messages
- Empty debug log file

**Possible Causes:**
- TT hardware not available
- `TT_METAL_HOME` not set
- Kernel compilation failed

**Solutions:**
```bash
# Check environment
echo $TT_METAL_HOME
echo $TT_METAL_DPRINT_CORES

# Try CPU simulation
export TT_METAL_SIMULATOR_EN=1
./debug_kernels_cpu.sh

# Check hardware
lspci | grep -i tenstorrent
```

### **Scenario 2: Kernels Start But No Computation**
**Symptoms:**
- See `KERNEL STARTED` messages
- No `Pixel(x,y)` calculation messages

**Possible Causes:**
- Tile processing loop issues
- Coordinate calculation errors
- Memory allocation problems

**Debug Steps:**
```bash
# Look for tile processing
cat kernel_debug.log | grep "Processing tile"

# Check coordinate setup
cat kernel_debug.log | grep "Coordinate"

# Verify tile counts
cat kernel_debug.log | grep "Num tiles"
```

### **Scenario 3: Computation But No Output**
**Symptoms:**
- See computation messages
- No `WRITER KERNEL` messages
- No data written to DRAM

**Possible Causes:**
- Circular buffer issues
- Dataflow kernel not launching
- Memory write problems

**Debug Steps:**
```bash
# Check for writer kernels
cat kernel_debug.log | grep "WRITER"

# Look for tile writing
cat kernel_debug.log | grep "Writing tile"

# Check circular buffer addresses
cat kernel_debug.log | grep "CB addr"
```

## 📈 **Advanced Debug Analysis**

### **Core-Specific Debugging**
```bash
# Debug only specific cores
export TT_METAL_DPRINT_CORES="0,0;1,1"  # Only cores (0,0) and (1,1)

# Debug only compute cores (TRISC)
export TT_METAL_DPRINT_CORES="trisc"

# Debug only dataflow cores (BRISC/NCRISC)
export TT_METAL_DPRINT_CORES="brisc,ncrisc"
```

### **Performance Analysis**
```bash
# Count debug messages per kernel type
cat kernel_debug.log | grep "COMPUTE KERNEL" | wc -l  # Should be 8
cat kernel_debug.log | grep "WRITER KERNEL" | wc -l   # Should be 8

# Check tile processing distribution
cat kernel_debug.log | grep "Processing tile" | head -20

# Verify all devices are working
cat kernel_debug.log | grep "Device ID" | sort | uniq
```

### **Coordinate Verification**
```bash
# Check coordinate ranges per device
cat kernel_debug.log | grep "Coordinate bounds"

# Verify pixel calculations
cat kernel_debug.log | grep "Pixel(" | head -10

# Check iteration counts (should vary across fractal)
cat kernel_debug.log | grep "iter=" | head -20
```

## 🎯 **Common Issues and Solutions**

### **Issue 1: "TT_METAL_HOME is not set"**
```bash
export TT_METAL_HOME="/home/tt-metal-apv"
```

### **Issue 2: "No kernel debug output"**
```bash
# Ensure DPRINT is enabled
export TT_METAL_DPRINT_CORES="all"
export TT_METAL_DPRINT_ENABLE=1

# Check if profiler is disabled
unset TT_METAL_PROFILER
```

### **Issue 3: "DPRINT and Profiler conflict"**
```bash
# Choose one or the other
unset TT_METAL_PROFILER          # For debugging
# OR
unset TT_METAL_DPRINT_CORES      # For profiling
```

### **Issue 4: "Hardware not available"**
```bash
# Use CPU simulation
export TT_METAL_SIMULATOR_EN=1
export TT_METAL_SIMULATOR_MODE=1
./debug_kernels_cpu.sh
```

## 📋 **Debug Checklist**

- ✅ `TT_METAL_HOME` set correctly
- ✅ `TT_METAL_DPRINT_CORES="all"` set
- ✅ `TT_METAL_PROFILER` unset
- ✅ Debug log file created
- ✅ See "KERNEL STARTED" messages (16 total: 8 compute + 8 writer)
- ✅ See "Device ID" 0-7 for all mesh devices
- ✅ See "Pixel(x,y)" calculation messages
- ✅ See "Writing tile" dataflow messages
- ✅ See "KERNEL COMPLETED" messages

## 🎉 **Success Indicators**

When debugging is working correctly, you should see:

1. **8 Compute Kernels** launching (one per mesh device)
2. **8 Writer Kernels** launching (one per mesh device)
3. **Device IDs 0-7** showing parallel execution
4. **Pixel calculations** with real complex coordinates
5. **Iteration counts** varying across the fractal
6. **Tile writing** to DRAM addresses
7. **Different coordinate ranges** per device (parallelization working)

This debug output will show you exactly what's happening in your kernels and help you understand the parallel execution across the 2×4 mesh! 🚀🔍
