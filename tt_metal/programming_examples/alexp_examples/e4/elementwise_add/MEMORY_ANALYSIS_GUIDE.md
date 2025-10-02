# Comprehensive Memory Analysis Guide

This guide provides in-depth tools and methodologies for analyzing L1 memory occupancy and circular buffer usage in TT-Metal programs, specifically for the distributed elementwise add implementations.

## 🎯 Overview

The customer version of distributed elementwise add experiences L1 memory pressure due to fixed circular buffer allocation that doesn't adapt to workload characteristics. This guide provides multiple analysis approaches to understand and diagnose such issues.

## 🛠️ Analysis Tools

### 1. Dynamic Memory Analysis (`memory_analysis_tool.py`)

**Purpose**: Simulates runtime memory allocation patterns and pressure analysis.

**Usage**:
```bash
# Analyze specific workload
python memory_analysis_tool.py 16 16

# Analyze different tile counts
python memory_analysis_tool.py 8 8
python memory_analysis_tool.py 32 32
```

**What it analyzes**:
- Circular buffer allocation strategies (fixed vs adaptive)
- Memory pressure metrics (CB miss rates, thrashing risk)
- Kernel memory footprint estimation
- Total L1 memory utilization

**Key Metrics**:
- **CB1 Miss Rate**: Percentage of B tiles that can't fit in CB1
- **Pressure Score**: 0-100 scale of memory pressure
- **Thrashing Risk**: LOW/MEDIUM/HIGH risk assessment
- **Memory Efficiency**: How well CB allocation matches workload needs

### 2. Static Code Analysis (`code_memory_analyzer.py`)

**Purpose**: Analyzes source code structure to extract actual memory allocation patterns.

**Usage**:
```bash
python code_memory_analyzer.py
```

**What it analyzes**:
- CircularBufferConfig declarations and sizes
- Kernel file sizes and complexity
- Allocation strategy detection (fixed vs adaptive)
- Compile-time constants and expressions

**Key Findings**:
- Source code differences between versions
- Actual CB allocation expressions
- Kernel complexity comparison
- Memory allocation strategy classification

### 3. Runtime Memory Profiling (`runtime_memory_profiler.py`)

**Purpose**: Analyzes actual TT-Metal runtime logs for real memory usage patterns.

**Usage**:
```bash
# With actual debug log
export TT_METAL_LOGGER_LEVEL=Debug
./your_program > debug.log 2>&1
python runtime_memory_profiler.py debug.log

# With sample data (demonstration)
python runtime_memory_profiler.py
```

**What it analyzes**:
- Actual CB allocations from runtime logs
- Memory events (reserve, push, pop operations)
- Kernel execution patterns
- Real pressure indicators from logs

### 4. Comprehensive Analysis Runner (`run_memory_analysis.py`)

**Purpose**: Runs all analysis tools together for complete picture.

**Usage**:
```bash
# Single workload analysis
python run_memory_analysis.py 16 16

# Multi-workload scaling analysis
python run_memory_analysis.py --multi

# Help
python run_memory_analysis.py --help
```

## 📊 Key Memory Concepts

### Wormhole L1 Memory Layout

```
┌─────────────────────────────────────┐ 0x0
│ System Reserved (~200KB)            │
├─────────────────────────────────────┤
│ Firmware & Runtime (~200KB)        │
├─────────────────────────────────────┤
│ Available for User Buffers          │ ← CB allocation happens here
│ - Circular Buffers (CB0-CB31)       │   (~1264KB available)
│ - L1 Buffers                        │
│ - Kernel Code/Data                  │
├─────────────────────────────────────┤
│ Stack & Local Variables             │
└─────────────────────────────────────┘ 1464KB (Total L1)
```

### Circular Buffer Pressure Analysis

**Customer Version Issues**:
```cpp
// Fixed allocation - inflexible
constexpr uint32_t cb0_tiles = 4;  // 16KB
constexpr uint32_t cb1_tiles = 4;  // 16KB
constexpr uint32_t cb2_tiles = 4;  // 16KB
constexpr uint32_t cb16_tiles = 4; // 16KB
// Total: 64KB regardless of workload
```

**Problems**:
- 16 B tiles need processing, but CB1 only holds 4 tiles
- CB1 miss rate = (16-4)/16 = 75%
- Constant CB1 thrashing and pressure

**Fixed Version Solution**:
```cpp
// Adaptive allocation - workload aware
if (tiles_per_shard > 16 || r_tiles > 16) {
    cb0_tiles = 2;  // Minimal for large workloads
    cb1_tiles = 2;
} else {
    cb0_tiles = std::min(6u, tiles_per_shard + 2);
    cb1_tiles = std::min(r_tiles, 6u);  // Up to 6 tiles for 16-tile case
}
```

**Benefits**:
- Better CB1 utilization for medium workloads
- Streaming mode for large workloads
- Conservative memory usage with safety margins

## 🔍 Diagnostic Workflow

### Step 1: Quick Analysis
```bash
python run_memory_analysis.py 16 16
```
Look for:
- High pressure scores (>70)
- MEDIUM/HIGH thrashing risk
- Large CB1 miss rates (>50%)

### Step 2: Multi-Workload Testing
```bash
python run_memory_analysis.py --multi
```
Look for:
- Scaling behavior across workload sizes
- Where pressure becomes critical
- Fixed vs customer version differences

### Step 3: Code Structure Analysis
```bash
python code_memory_analyzer.py
```
Look for:
- Fixed vs adaptive allocation strategies
- Actual CB size expressions
- Kernel complexity differences

### Step 4: Runtime Validation
```bash
# Enable debug logging
export TT_METAL_LOGGER_LEVEL=Debug

# Run actual program
./distributed_elementwise_add_customer 16 > customer_debug.log 2>&1
./distributed_elementwise_add_fixed 16 > fixed_debug.log 2>&1

# Analyze logs
python runtime_memory_profiler.py customer_debug.log
python runtime_memory_profiler.py fixed_debug.log
```

## 📈 Understanding the Results

### Memory Pressure Indicators

| Metric | Low Pressure | Medium Pressure | High Pressure |
|--------|-------------|----------------|---------------|
| CB1 Miss Rate | <25% | 25-75% | >75% |
| Pressure Score | <30 | 30-70 | >70 |
| Thrashing Risk | LOW | MEDIUM | HIGH |
| CB Utilization | >80% | 40-80% | <40% |

### Workload Classification

| Workload Type | Tile Count | Strategy | Expected Behavior |
|---------------|------------|----------|-------------------|
| Small | ≤8 tiles | Optimal buffering | High performance, low pressure |
| Medium | 9-16 tiles | Balanced allocation | Good performance, some pressure |
| Large | 17-32 tiles | Conservative allocation | Stable, higher pressure |
| Very Large | >32 tiles | Streaming mode | Slower but stable |

## 🛠️ Debugging Tips

### Enable TT-Metal Memory Reporting
```bash
export TT_METAL_MEMORY_REPORT=1
export TT_METAL_LOGGER_LEVEL=Debug
```

### Common Pressure Symptoms
- Long kernel execution times
- High CB reservation wait times
- Numerical accuracy issues
- Inconsistent performance across runs

### Optimization Strategies
1. **Adaptive CB sizing** based on workload
2. **Producer-consumer coordination** in kernels
3. **Streaming mode** for large workloads
4. **Conservative memory targets** with safety margins

## 🎯 Best Practices

### For Development
1. Always test with multiple workload sizes
2. Monitor L1 memory usage during development
3. Use adaptive allocation strategies
4. Include debug output for CB operations

### For Production
1. Profile memory usage with representative workloads
2. Set conservative memory targets
3. Implement fallback strategies for large workloads
4. Monitor for memory pressure indicators

### For Debugging
1. Enable comprehensive logging
2. Use static analysis to understand allocation patterns
3. Compare runtime behavior against expected patterns
4. Validate numerical accuracy under memory pressure

## 📚 Additional Resources

- **TT-Metal Memory Documentation**: `/tech_reports/memory/allocator.md`
- **Wormhole Architecture Guide**: Hardware specifications and L1 layout
- **Circular Buffer API**: TT-Metal programming guide
- **Performance Profiling**: Tracy integration for real-time monitoring

## 🔧 Tool Extensions

The provided tools can be extended for:
- **Custom workload patterns**: Modify analysis parameters
- **Different architectures**: Add Blackhole/Grayskull support
- **Real-time monitoring**: Integration with Tracy profiler
- **Automated testing**: CI/CD integration for memory regression testing

This comprehensive analysis framework provides the foundation for understanding, diagnosing, and optimizing L1 memory usage in TT-Metal programs.
