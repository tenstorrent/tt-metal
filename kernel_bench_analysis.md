# Kernel Bench: Comprehensive Repository Analysis

**Repository**: [tenstorrent/kernel_bench](https://github.com/tenstorrent/kernel_bench)
**Analysis Date**: 2026-04-05

---

## 1. Purpose and Overview

Kernel Bench is a **competitive leaderboard and evaluation platform** for AI-driven code generation of TT-NN kernel implementations. The core idea: given a PyTorch operation (e.g., `torch.cosh`), can an AI tool autonomously generate a correct TTNN implementation (`ttnn_cosh_impl.py`) that passes rigorous numerical accuracy tests on real Tenstorrent hardware?

The repository tracks community submissions from different AI-based code generation tools, automatically runs evaluations on Tenstorrent N300 hardware, and maintains a live leaderboard in the README.

**Key metrics**:
- **42 benchmark operations** defined (from simple unary activations to complex reductions and data movement)
- **103 total submissions** across all operations
- **3 status tiers**: Solved (100% pass rate), In Progress (partial passes), Failed (all tests fail)
- **11 contributors** (mix of Tenstorrent engineers and external participants)

---

## 2. Repository Structure

```
kernel_bench/
├── README.md                              # Auto-generated leaderboard
├── pyproject.toml                         # Python project config (uv-managed)
│
├── benchmark/                             # Test definitions (42 operations)
│   └── {operation}/
│       ├── {operation}_test_generator.py  # PyTorch reference + test cases
│       └── task.md                        # Natural language task description
│
├── submissions/                           # All submissions + results
│   └── {operation}/
│       ├── {submission_name}/
│       │   ├── tool.yaml                 # Tool config (how to generate code)
│       │   ├── eval_results.json         # Evaluation results
│       │   ├── docs/                     # Reference docs for the AI tool
│       │   ├── templates/                # Code templates
│       │   └── result/                   # Generated code
│       │       ├── 1/, 2/, ..., N/       # Numbered generation attempts
│       │       └── latest -> N           # Symlink to latest attempt
│       └── {operation}_submissions.md    # Per-operation leaderboard
│
├── documents/tm/                          # Tenstorrent Metal documentation
│   ├── docs/                             # API references (SFPU, compute, data movement, etc.)
│   ├── templates/                        # Working reference implementations
│   │   ├── singlecore_transpose/
│   │   ├── multicore_transpose/
│   │   ├── singlecore_slice/
│   │   └── multicore_slice/
│   ├── recipe_prompt.txt                 # Master prompt template
│   ├── generate_recipe_prompt.txt
│   └── prepare_recipe_prompt.txt
│
├── tools/
│   ├── eval/                             # Evaluation infrastructure
│   │   ├── eval.py                       # Unified CLI entry point
│   │   ├── eval_engine.py                # Test execution engine
│   │   ├── test_generator_base.py        # Base class for test generators
│   │   ├── data_aggregator.py            # Results collection
│   │   ├── op_report_generator.py        # Per-operation reports
│   │   └── repo_report_generator.py      # Main README update
│   └── codegen/                          # Code generation orchestration
│       ├── codegen.py                    # Tool runner / submission analyzer
│       ├── tool_registry.py              # Tool handler registry
│       ├── tools/                        # Tool implementations
│       │   ├── replay_tool.py            # Replay-based code gen
│       │   ├── manual_tool.py            # Manual code gen
│       │   └── bash_tool.py              # Script-based code gen
│       └── utilities/
│
└── .github/workflows/                    # CI/CD automation
    ├── dispatch-submissions.yml          # Auto-trigger on push to submissions/
    ├── single-submission-codegen.yml     # Run code gen for one submission
    ├── single-submission-eval.yml        # Run eval for one submission
    ├── run-evaluations.yml               # Batch evaluate all operations
    └── test-tool-revision.yml            # Test tool changes
```

---

## 3. Operation Categories

### 3.1 All 42 Benchmark Operations

| Category | Operations | Count |
|----------|-----------|-------|
| **Unary Activations** | hardsigmoid, hardswish, hardtanh, sigmoid, softsign, softplus, selu, swish, hardshrink, softshrink | 10 |
| **Unary Math** | cosh, sinh, atanh, cbrt, digamma, lgamma, frac, rpow, multigammaln, polygamma, logical_not | 11 |
| **Reductions** | sum, prod, mean, std, max, min, argmax, topk, cumsum, cumprod, reductions (composite) | 11 |
| **Data Movement** | transpose, slice, pad, concat, permute, reshape, tril, triu | 8 |
| **Gated Units** | geglu, glu, reglu, swiglu | 4 |
| **Multi-operand** | matmul | 1 |

### 3.2 Leaderboard Status (as of last update)

| Status | Operations | Breakdown |
|--------|-----------|-----------|
| **Solved** (100% pass) | 3 | hardsigmoid, hardswish, softsign |
| **Near** (97%+) | 1 | hardtanh |
| **In Progress** (>0% pass) | 15 | sigmoid, logical_not, softplus, cosh, sinh, selu, digamma, frac, swish, hardshrink, atanh, rpow, softshrink, lgamma, tril |
| **Failed** (0% pass) | 20 | slice, transpose, cbrt, pad, cumprod, geglu, glu, max, multigammaln, polygamma, sum, triu, matmul, argmax, cumsum, mean, min, prod, reglu, std, swiglu, topk |
| **Unattacked** | 3 | concat, permute, reshape |

**Key insight**: Simple unary activations (hardsigmoid, hardswish, softsign) have been solved. Complex operations like reductions, data movement, and multi-operand operations remain largely unsolved.

---

## 4. Evaluation System

### 4.1 Test Generator Framework

Every operation has a test generator class (`BaseTestGenerator` subclass) that defines:

1. **`reference_impl()`** - PyTorch reference implementation (ground truth)
2. **`generate_test_params()`** - Comprehensive test cases covering:
   - Multiple tensor ranks (0D scalar through 4D batched)
   - Multiple dtypes (`float32`, `bfloat16`)
   - Edge cases (zeros, small values, large values, boundary shapes)
   - Shape variety (tile-aligned and non-aligned: 31, 32, 33, 64, 127, 128, etc.)

Example: The `cosh` test generator produces ~150+ test cases across 5 ranks, 2 dtypes, and 5 value distributions (standard, small, large-positive, large-negative, zeros).

### 4.2 Accuracy Metric: ULP-Based Quantization Error

The validation system uses a sophisticated **quantization-aware epsilon comparison**:

1. Computes the expected quantization error for the target dtype (bfloat16 has 7 mantissa bits, float16 has 10)
2. Calculates error ratio: `|reference - ttnn| / quantization_epsilon`
3. Converts to accuracy on a logarithmic scale:
   - Error < 1 ULP: accuracy = 1.0
   - Error < 2 ULP: accuracy = 0.99
   - Error < 4 ULP: accuracy = 0.98
   - And so on...
4. A test **passes** if accuracy >= 0.99 (within ~2 ULP)

This approach correctly handles the tile-format precision constraints of Tenstorrent hardware, where bfloat16 is the dominant compute dtype.

### 4.3 Evaluation Pipeline

```
Test Generator → Generate Params → Load TTNN Impl → Convert to TTNN tensors
                                                   → Run on device
                                                   → Compare vs PyTorch reference
                                                   → Save eval_results.json
```

The evaluation runs on actual Tenstorrent hardware (`tt-ubuntu-2204-n300-stable` runners) using the official `tt-metalium` container image.

---

## 5. Code Generation System

### 5.1 Submission Structure

Each submission consists of:

- **`tool.yaml`**: Declares the code generation tool and its configuration
- **`docs/`**: API documentation provided as context to the AI tool
- **`templates/`**: Working reference code examples
- **`result/`**: Generated code outputs (numbered attempts)

### 5.2 Tool Types

The tool registry supports three generation approaches:

| Tool | Description | Example |
|------|-------------|---------|
| **`bash`** | Runs a shell script (`script.sh`) to generate code | Most submissions use this |
| **`replay`** | Replays a recorded code generation session | Used for reproducibility |
| **`manual`** | Manual code placement | Baseline submissions |

A typical `tool.yaml`:
```yaml
tool: bash
runner: wormhole
with:
  - script: script.sh
  - timeout: 600
```

The `bash` tool is most common: the `script.sh` typically invokes Claude Code CLI (`claude`) with a recipe prompt that includes the task description, API docs, and template code.

### 5.3 Code Generation Flow

The codegen workflow installs Claude Code CLI via npm, then:
1. Reads `tool.yaml` from the submission directory
2. Dispatches to the appropriate tool handler
3. The tool generates `ttnn_{operation}_impl.py` in `result/N/code/`
4. Updates the `latest` symlink
5. Commits and pushes the generated code
6. Automatically triggers evaluation

### 5.4 Recipe Prompts

The `documents/tm/recipe_prompt.txt` provides a structured prompt template that includes:
- Architecture context (Tensix cores, tile-based computing, NoC)
- API documentation (SFPU, compute, data movement, circular buffers)
- Working code templates (singlecore/multicore transpose and slice implementations)
- Explicit requirements (multi-core, n-dimensional, row-major interleaved)

---

## 6. CI/CD Automation

### 6.1 Workflow Architecture

```
Push to submissions/ ──→ dispatch-submissions.yml
                              │
                    ┌─────────┴─────────┐
                    ▼                   ▼
        Code changes?              Result changes?
        (tool.yaml, docs,          (result/*/code/)
         templates, scripts)
                    │                   │
                    ▼                   ▼
    single-submission-codegen.yml   single-submission-eval.yml
    (generates code on HW)         (evaluates on HW)
                    │
                    └──→ triggers eval ──→ single-submission-eval.yml
                                                    │
                                                    ▼
                                          eval_results.json
                                          + report generation
```

### 6.2 Batch Evaluation

`run-evaluations.yml` provides a parallel evaluation pipeline:
1. **Discover**: Dynamically finds all operations with submissions
2. **Create branch**: Shared branch for results
3. **Run evaluations**: Matrix strategy - one parallel job per operation on N300 hardware
4. **Combine results**: Downloads artifacts, regenerates reports, creates PR

### 6.3 Hardware Requirements

All evaluations run on:
- **Runner**: `tt-ubuntu-2204-n300-stable` (Tenstorrent N300 card)
- **Container**: `ghcr.io/tenstorrent/tt-metal/tt-metalium/ubuntu-22.04-dev-amd64:latest`
- **Device access**: `/dev/tenstorrent` with hugepages support
- Device is reset between submissions (`tt-smi -r 0`)

---

## 7. Submission Ecosystem

### 7.1 Submission Distribution

Top operations by submission count:
| Operation | Submissions | Best Pass Rate |
|-----------|-------------|----------------|
| cosh | 10 | 68.7% |
| sinh | 6 | 68.7% |
| atanh | 6 | 53.8% |
| hardsigmoid | 3 | 100% (Solved) |
| hardswish | 3 | 100% (Solved) |
| softsign | 3 | 100% (Solved) |
| matmul | 3 | Failed |

### 7.2 Leading Contributors

| Contributor | Approach | Solved Ops |
|-------------|----------|------------|
| **nachiket-adaptive-luts** | Adaptive LUT-based approximations | hardsigmoid, hardswish, softsign (shared) |
| **rlesliehurd-aiworkflow** | AI workflow automation | hardsigmoid, hardswish, softsign (shared) |
| **jmalone-tt** | Claude Code + doc/template-based prompting | tril (partial) |
| **ayerofieiev-tt** | Claude Code integration | Various in-progress |

### 7.3 Success Patterns

**What works** (Solved operations):
- Simple unary activations with known closed-form SFPU implementations
- Operations that map directly to existing TTNN primitives
- Submissions with good reference documentation and templates

**What fails** (Unsolved operations):
- Reductions requiring cross-tile communication (sum, mean, prod, argmax)
- Data movement operations requiring complex memory patterns (transpose, slice, pad)
- Multi-operand operations (matmul) requiring FPU matrix engine usage
- Gated activation units requiring multiple tensor inputs (geglu, glu, swiglu)

---

## 8. Technical Documentation Bundle

The repository includes a curated documentation package in `documents/tm/`:

### 8.1 API Documentation
| Document | Coverage |
|----------|----------|
| `kernel_api_sfpu.md` | SFPI programming interface, vector operations, conditional execution |
| `kernel_api_compute_operations.md` | FPU/SFPU compute API |
| `kernel_api_data_movement.md` | NoC read/write, DRAM/L1 transfers |
| `kernel_api_circular_buffers.md` | CB management (reserve, push, wait) |
| `kernel_api_common.md` | Common kernel utilities |
| `kernel_api_packer.md` | Pack/unpack operations |
| `host_api_py.md` | Python host API reference |
| `handling_inf_and_none.md` | Special value handling |

### 8.2 Working Templates
- **Singlecore transpose**: Complete reader/writer kernels + Python host code
- **Multicore transpose**: Parallelized version with core distribution
- **Singlecore slice**: 4D tensor slicing with TensorAccessor
- **Multicore slice**: Parallelized 4D slicing

These templates serve as few-shot examples for AI code generation tools.

---

## 9. Expected Submission Output

A valid submission must produce:

```python
# File: result/latest/code/ttnn_{operation}_impl.py

def ttnn_{operation}(*args, **kwargs):
    """TTNN implementation matching PyTorch signature"""
    # Implementation using tt_metal APIs:
    # 1. Set up program, kernels, circular buffers
    # 2. Configure reader/compute/writer kernels
    # 3. Execute on device
    # 4. Return ttnn.Tensor result
    return result_tensor
```

The function receives TTNN tensors (already on device, in TILE_LAYOUT) and must return a TTNN tensor. The evaluation framework handles the PyTorch-to-TTNN conversion.

---

## 10. Key Observations and Insights

### 10.1 Difficulty Gradient

The benchmark reveals a clear difficulty hierarchy for AI code generation on Tenstorrent hardware:

1. **Easy** (Solved): Unary pointwise activations (hardsigmoid, hardswish, softsign) - these map to simple SFPU kernel chains
2. **Medium** (In Progress): Unary math with numerical challenges (cosh, sinh, atanh) - correct but precision issues
3. **Hard** (Failed): Operations requiring non-trivial data movement patterns (transpose, slice, pad)
4. **Very Hard** (Failed): Reductions, multi-operand ops (matmul, sum, argmax) - require cross-core coordination

### 10.2 Precision is the Main Barrier

Most in-progress operations generate functionally correct code but fail accuracy thresholds. The ULP-based accuracy metric with a 0.99 threshold (requiring <2 ULP error) is demanding. Operations like cosh (68.7% pass rate) likely fail on edge cases (large inputs, near-zero) where bfloat16 precision is insufficient without careful approximation strategies.

### 10.3 Platform for AI Tool Benchmarking

The repository effectively serves as a benchmark for comparing AI coding assistants on hardware-specific kernel development:
- Can the AI understand tile-based memory layouts?
- Can it generate correct circular buffer management?
- Can it handle the reader/compute/writer kernel decomposition?
- Can it produce numerically stable implementations within bfloat16 precision?

### 10.4 Infrastructure Maturity

The evaluation infrastructure is well-engineered:
- Fully automated CI/CD with parallel evaluation on real hardware
- Deterministic test generation with fixed seeds (`KERNEL_BENCH_SEED=42`)
- Quantization-aware accuracy metrics appropriate for bfloat16/float16 hardware
- Per-submission result tracking with generation attempt counting
- Auto-generated leaderboard and per-operation reports

---

## 11. Languages and Dependencies

| Language | Size | Usage |
|----------|------|-------|
| C++ | 6.05 MB | Kernel code (reader/writer/compute), API docs, templates |
| Python | 5.28 MB | Test generators, evaluation engine, host-side orchestration, generated TTNN implementations |
| TypeScript | 352 KB | Web frontend (likely a leaderboard dashboard) |
| JavaScript | 22 KB | Web support |
| Shell | 2 KB | Build/setup scripts |
| CMake | 1 KB | Build configuration |
| CSS | 488 B | Web styling |

**Python dependencies**: `torch 2.7.1`, `ttnn`, `numpy`, `pytest`, `pyyaml`, `loguru`
**Build tool**: `uv` (fast Python package manager)
**Runtime**: Python 3.10 on Ubuntu 22.04

---

## 12. Summary

Kernel Bench is an automated competitive platform that measures how well AI code generation tools can produce correct, hardware-specific TT-NN kernel implementations. With 42 operations spanning unary activations to complex reductions and data movement, it provides a graded difficulty benchmark. Currently, only 3 operations are fully solved (simple activations), while 15 are partially working and 20+ remain unsolved. The repository's well-designed CI/CD pipeline, quantization-aware accuracy metrics, and comprehensive test coverage make it a rigorous evaluation framework for AI-assisted kernel development on Tenstorrent hardware.
