# TT-Metal Operator OOM (Out of Memory) Comprehensive Analysis

**Generated:** 2025-08-18 23:17:59
**Focus:** Out of Memory failures across all operators
**Scope:** Analysis of 46 operators with 1322 OOM failures

## 🚨 OOM EXECUTIVE SUMMARY

| **Metric** | **Value** | **Impact** |
|------------|-----------|------------|
| **Total OOM Failures** | 1,322 | Critical |
| **Operators Affected** | 44/46 | 95.7% |
| **Unique Problem Shapes** | 146 | Memory Patterns |
| **Critical Shapes (>1GB)** | 55 | Immediate Fix Needed |
| **Large Shapes (500MB-1GB)** | 112 | High Priority |
| **Medium Shapes (100-500MB)** | 412 | Medium Priority |
| **Small Shapes (<100MB)** | 743 | Low Priority |

## 🔥 CRITICAL OOM SHAPES (>1GB)

| **Shape** | **Memory (GB)** | **Memory (MB)** | **Operators Affected** | **Frequency** |
|-----------|-----------------|-----------------|------------------------|---------------|
| `[1, 96, 2176, 3840]` | **1.49GB** | 1530.0MB | split, split_with_sizes | 2 |
| `[1, 192, 2176, 1920]` | **1.49GB** | 1530.0MB | split_with_sizes | 1 |
| `[12, 8160, 8160]` | **1.49GB** | 1524.0MB | softmax, bmm | 2 |
| `[1, 12, 8160, 8160]` | **1.49GB** | 1524.0MB | softmax | 1 |
| `[1, 96, 2160, 3840]` | **1.48GB** | 1518.8MB | log, div, clampmin... | 41 |
| `[1, 96, 4320, 1920]` | **1.48GB** | 1518.8MB | gelu, div, mean... | 8 |

## 📊 OOM FAILURES BY OPERATOR

| **Operator** | **OOM Count** | **Worst Shape** | **Max Memory** | **Status** |
|--------------|---------------|-----------------|----------------|------------|
| **transpose** | 40 | `[1, 96, 2160, 3840]` | **1.48GB** | 🚨 CRITICAL |
| **div** | 38 | `[1, 96, 2160, 3840]` | **1.48GB** | 🚨 CRITICAL |
| **hardtanh** | 37 | `[1, 96, 2160, 3840]` | **1.48GB** | 🚨 CRITICAL |
| **leakyrelu** | 37 | `[1, 96, 2160, 3840]` | **1.48GB** | 🚨 CRITICAL |
| **maxpool2d** | 37 | `[1, 96, 2160, 3840]` | **1.48GB** | 🚨 CRITICAL |
| **mul** | 37 | `[1, 96, 2160, 3840]` | **1.48GB** | 🚨 CRITICAL |
| **native_batch_norm** | 37 | `[1, 96, 2160, 3840]` | **1.48GB** | 🚨 CRITICAL |
| **splitwithsizes** | 37 | `[1, 96, 2160, 3840]` | **1.48GB** | 🚨 CRITICAL |
| **add** | 36 | `[1, 96, 2160, 3840]` | **1.48GB** | 🚨 CRITICAL |
| **sub** | 36 | `[1, 96, 2160, 3840]` | **1.48GB** | 🚨 CRITICAL |
| **cat** | 35 | `[1, 96, 2160, 3840]` | **1.48GB** | 🚨 CRITICAL |
| **mish** | 35 | `[1, 96, 2160, 3840]` | **1.48GB** | 🚨 CRITICAL |
| **relu** | 35 | `[1, 96, 2160, 3840]` | **1.48GB** | 🚨 CRITICAL |
| **softplus** | 35 | `[1, 96, 2160, 3840]` | **1.48GB** | 🚨 CRITICAL |
| **clone** | 34 | `[1, 96, 2160, 3840]` | **1.48GB** | 🚨 CRITICAL |
| **concat** | 34 | `[1, 96, 2160, 3840]` | **1.48GB** | 🚨 CRITICAL |
| **identity** | 34 | `[1, 96, 2160, 3840]` | **1.48GB** | 🚨 CRITICAL |
| **permute** | 34 | `[1, 96, 2160, 3840]` | **1.48GB** | 🚨 CRITICAL |
| **sigmoid** | 34 | `[1, 96, 2160, 3840]` | **1.48GB** | 🚨 CRITICAL |
| **silu** | 34 | `[1, 96, 2160, 3840]` | **1.48GB** | 🚨 CRITICAL |
| **tanh** | 34 | `[1, 96, 2160, 3840]` | **1.48GB** | 🚨 CRITICAL |
| **unsafeview** | 33 | `[1, 96, 2160, 3840]` | **1.48GB** | 🚨 CRITICAL |
| **unsqueeze** | 33 | `[1, 96, 2160, 3840]` | **1.48GB** | 🚨 CRITICAL |
| **view** | 33 | `[1, 96, 2160, 3840]` | **1.48GB** | 🚨 CRITICAL |
| **clampmin** | 31 | `[1, 96, 2160, 3840]` | **1.48GB** | 🚨 CRITICAL |
| **expand** | 31 | `[1, 96, 2160, 3840]` | **1.48GB** | 🚨 CRITICAL |
| **linalgvectornorm** | 31 | `[1, 96, 2160, 3840]` | **1.48GB** | 🚨 CRITICAL |
| **silu_inplace** | 31 | `[1, 96, 2160, 3840]` | **1.48GB** | 🚨 CRITICAL |
| **stack** | 30 | `[1, 96, 2160, 3840]` | **1.48GB** | 🚨 CRITICAL |
| **gelu** | 27 | `[1, 96, 2160, 3840]` | **1.48GB** | 🚨 CRITICAL |
| **ones** | 27 | `[1, 96, 2160, 3840]` | **1.48GB** | 🚨 CRITICAL |
| **copy** | 26 | `[1, 96, 2160, 3840]` | **1.48GB** | 🚨 CRITICAL |
| **dropout** | 26 | `[1, 96, 2160, 3840]` | **1.48GB** | 🚨 CRITICAL |
| **softmax** | 26 | `[1, 12, 8160, 8160]` | **1.49GB** | 🚨 CRITICAL |
| **upsample_nearest2d** | 24 | `[1, 96, 1088, 1920]` | **0.37GB** | 🟡 MEDIUM |
| **clamp** | 23 | `[1, 96, 2160, 3840]` | **1.48GB** | 🚨 CRITICAL |
| **mean** | 23 | `[1, 96, 2160, 3840]` | **1.48GB** | 🚨 CRITICAL |
| **topk** | 23 | `[1, 96, 2160, 3840]` | **1.48GB** | 🚨 CRITICAL |
| **geluactivation** | 22 | `[1, 96, 2160, 3840]` | **1.48GB** | 🚨 CRITICAL |
| **split** | 19 | `[1, 96, 2176, 3840]` | **1.49GB** | 🚨 CRITICAL |
| **split_with_sizes** | 19 | `[1, 192, 2176, 1920]` | **1.49GB** | 🚨 CRITICAL |
| **bmm** | 14 | `[12, 8160, 8160]` | **1.49GB** | 🚨 CRITICAL |
| **log** | 13 | `[1, 96, 2160, 3840]` | **1.48GB** | 🚨 CRITICAL |
| **addmm** | 7 | `[64000, 512]` | **0.06GB** | 🟢 LOW |

## 🔍 DETAILED SHAPE ANALYSIS

### Most Problematic Shapes (by operator count)

| **Shape** | **Memory** | **Operators Affected** | **Total Failures** |
|-----------|------------|------------------------|---------------------|
| `[1]` | **0.00GB** | 44 ops | 44 failures |
| `[1, 96, 2160, 3840]` | **1.48GB** | 41 ops | 41 failures |
| `[1, 192, 1080, 1920]` | **0.74GB** | 41 ops | 41 failures |
| `[1, 384, 540, 960]` | **0.37GB** | 41 ops | 41 failures |
| `[1, 768, 270, 480]` | **0.19GB** | 41 ops | 41 failures |
| `[1, 1536, 135, 240]` | **0.09GB** | 41 ops | 41 failures |
| `[1, 96, 1080, 1920]` | **0.37GB** | 40 ops | 40 failures |
| `[1, 3, 4320, 7680]` | **0.19GB** | 40 ops | 40 failures |
| `[1, 192, 540, 960]` | **0.19GB** | 40 ops | 40 failures |
| `[1, 384, 270, 480]` | **0.09GB** | 40 ops | 40 failures |
| `[1, 96, 2160, 1920]` | **0.74GB** | 38 ops | 38 failures |
| `[1, 192, 1080, 960]` | **0.37GB** | 38 ops | 38 failures |
| `[1, 384, 540, 480]` | **0.19GB** | 38 ops | 38 failures |
| `[1, 768, 270, 240]` | **0.09GB** | 38 ops | 38 failures |
| `[1, 96, 540, 960]` | **0.09GB** | 38 ops | 38 failures |
| `[1, 96, 1088, 1920]` | **0.37GB** | 32 ops | 32 failures |
| `[1, 192, 544, 960]` | **0.19GB** | 31 ops | 31 failures |
| `[1, 384, 272, 480]` | **0.09GB** | 31 ops | 31 failures |
| `[1, 768, 135, 240]` | **0.05GB** | 31 ops | 31 failures |
| `[1, 3, 2176, 3840]` | **0.05GB** | 30 ops | 30 failures |

## 📈 MEMORY DISTRIBUTION ANALYSIS

### OOM Failures by Memory Category

| **Category** | **Range** | **Count** | **Percentage** | **Operators** |
|--------------|-----------|-----------|----------------|---------------|
| **SMALL** | < 100MB | 743 | 56.2% | 44 |
| **MEDIUM** | 100MB - 500MB | 412 | 31.2% | 43 |
| **LARGE** | 500MB - 1GB | 112 | 8.5% | 41 |
| **CRITICAL** | > 1GB | 55 | 4.2% | 42 |

## 🎯 8K RESOLUTION IMPACT ANALYSIS

**8K-Related OOM Failures:** 614/1322 (46.4%)

### Top 8K Resolution OOM Issues

| **Shape** | **Memory** | **Operator** | **8K Context** |
|-----------|------------|--------------|----------------|
| `[1, 96, 2176, 3840]` | **1.49GB** | split | 8K derivative |
| `[1, 192, 2176, 1920]` | **1.49GB** | split_with_sizes | 8K derivative |
| `[1, 96, 2176, 3840]` | **1.49GB** | split_with_sizes | 8K derivative |
| `[1, 96, 2160, 3840]` | **1.48GB** | add | Half 8K |
| `[1, 96, 2160, 3840]` | **1.48GB** | cat | Half 8K |
| `[1, 96, 2160, 3840]` | **1.48GB** | clamp | Half 8K |
| `[1, 96, 4320, 1920]` | **1.48GB** | clamp | 8K derivative |
| `[1, 96, 2160, 3840]` | **1.48GB** | clampmin | Half 8K |
| `[1, 96, 2160, 3840]` | **1.48GB** | clone | Half 8K |
| `[1, 96, 2160, 3840]` | **1.48GB** | concat | Half 8K |
| `[1, 96, 2160, 3840]` | **1.48GB** | copy | Half 8K |
| `[1, 96, 4320, 1920]` | **1.48GB** | copy | 8K derivative |
| `[1, 96, 2160, 3840]` | **1.48GB** | div | Half 8K |
| `[1, 96, 4320, 1920]` | **1.48GB** | div | 8K derivative |
| `[1, 96, 2160, 3840]` | **1.48GB** | dropout | Half 8K |

## 💡 OOM RESOLUTION RECOMMENDATIONS

### 🚨 IMMEDIATE CRITICAL ACTIONS

1. **Critical Memory Issues (>1GB) - URGENT:**
   1. **add** operator - Max: 1.48GB
   2. **bmm** operator - Max: 1.49GB
   3. **cat** operator - Max: 1.48GB
   4. **clamp** operator - Max: 1.48GB
   5. **clampmin** operator - Max: 1.48GB

2. **Technical Solutions:**
   - **Tensor Chunking:** Implement automatic splitting for tensors >500MB
   - **Memory Streaming:** Process large tensors in streaming fashion
   - **Precision Optimization:** Use mixed precision (fp16/bf16) strategically
   - **Memory Pooling:** Implement efficient memory reuse patterns
   - **Progressive Testing:** Start with smaller shapes, scale up gradually

3. **Device Configuration:**
   - **Memory Limits:** Review and potentially increase device memory allocation
   - **Fallback Mechanisms:** Implement CPU fallback for oversized tensors
   - **Memory Monitoring:** Add real-time memory usage tracking

### 📋 PRIORITIZED ACTION PLAN

#### Phase 1: Critical (Immediate - 1-2 weeks)
- Fix operators with >1GB tensor failures
- Implement tensor chunking for shapes >500MB
- Add memory limit checks before operation execution

#### Phase 2: High Priority (2-4 weeks)
- Optimize operators with 500MB-1GB failures
- Implement memory streaming for large operations
- Add progressive shape testing framework

#### Phase 3: Medium Priority (1-2 months)
- Address 100-500MB memory issues
- Optimize memory usage across all operators
- Implement comprehensive memory profiling

---

*OOM Analysis Report generated by TT-Metal Memory Analyzer*
*Timestamp: 2025-08-18 23:17:59*
