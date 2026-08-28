# KDA theoretical performance model — development specification

Status: approved for local implementation on 2026-08-28
Scope: KDA only; no generic performance-model changes
Branch: momcilo/kda_perf_model
Base: origin/kda-split/04b-recurrent-chunk-scan at ee7353a69eb38a2da9e059b276cf546c0dc46811
Tracking: tt-metal_tracker-858.4.1

## 1. Goal

Add a KDA-local theoretical performance model for every operation introduced by
PR1 through PR6:

1. sigmoid gated RMS norm
2. QKV causal conv1d with SiLU
3. reduce affine transforms
4. affine exclusive scan
5. prepare chunk recurrence
6. recurrent chunk scan
7. recurrent chunk scan summary

The model supplies lower bounds to the Tracy operation profiler from C++ and
the same calculations to the realtime-profiler performance tests from Python.

The model answers this question:

> Given the canonical mathematical operator graph, tensor placement, documented
> hardware capabilities, and the post-harvest device, what is its theoretical
> hardware lower-bound time?

It does not predict the current implementation's runtime.

## 2. Modeling contract

For each operation:

1. Derive total work from the operation's mathematical/PyTorch oracle.
2. Assume work is split perfectly and equally across every available Tensix
   core on the device's post-harvest grid.
3. Divide each documented class of work by its aggregate hardware capability.
4. Count mandatory DRAM traffic as one physical read of every unique DRAM input
   and one physical write of every unique DRAM output.
5. Assume compute and DRAM transfer overlap perfectly.

Work is the structural operator count of the canonical oracle, not an arbitrary
algebraic rewrite and not a value-dependent specialization. Neutral
initializers used only to define a fold are eliminated, an n-operand sum has
n - 1 additions, and work producing no observable output is eliminated.
Operators whose actual tensor operands happen to be zero, identity, or
disjoint-support are otherwise counted. Named SFPU primitives are the atomic
exception described in section 3.3.

The result is:

    ideal_fpu_ns = ceil(ideal_fpu_cycles * 1000 / clock_mhz)
    ideal_dram_ns = ceil(mandatory_dram_bytes / 512)
    ideal_ns = max(ideal_fpu_ns, ideal_dram_ns)

Here clock_mhz is integer megahertz and 512 is aggregate bytes/ns.

### 2.1 Intentionally ignored implementation details

The model must not inspect or encode:

- program grids, active-core selections, or busiest-core work
- work splitting, scheduling, kernel loops, or scan-tree structure
- LLK sequences, tilize/untilize, pack/unpack, or implementation copies
- circular-buffer sizes, L1 intermediates, synchronization, or NoC traffic
- repeated DRAM reads/writes caused by the current implementation
- current kernel decomposition of a mathematical expression

No KDA program factory, kernel source, or scheduling helper is a source of truth
for work totals.

## 3. Hardware capabilities

### 3.1 Aggregate core count and clock

At runtime:

    core_count =
        device->compute_with_storage_grid_size().x *
        device->compute_with_storage_grid_size().y

This uses every available core on the post-harvest grid, including non-square
grids. It does not use the operation's selected core range. The model is
per-chip; KDA does not expose a multi-chip operation in this scope.

Use device->get_clock_rate_mhz() for clock frequency.

The initial implementation supports Blackhole. For another architecture, an
unexpected non-device tensor, invalid fidelity, arithmetic overflow, or failed
narrowing, return a zero-valued KDA estimate and emit a warning rather than
aborting profiling or silently applying invalid values. The required generic
profiler wrapper remains at its safe one-cycle/one-nanosecond minimum.

### 3.2 Documented FPU work classes

The following per-core rates are the only compute capabilities modeled:

| Mathematical work | Capability per core per cycle | Fidelity |
| --- | ---: | --- |
| Dense matrix multiplication | 4096 FLOPs = 2048 MACs | divide by 1, 2, 3, or 4 for LoFi, HiFi2, HiFi3, or HiFi4 |
| Elementwise add/subtract | 128 result elements | no scaling |
| Elementwise multiply, square, scalar multiply, broadcast multiply | 128 result elements | same fidelity scaling |
| Sum/average reduction | 256 input elements | same fidelity scaling |

Work classes are additive because they consume the documented matrix/FPU issue
capacity. Calculate the final ceiling exactly with a common integer denominator:

    cycle_numerator =
        dense_flops * fidelity_factor +
        32 * multiply_results * fidelity_factor +
        32 * add_results +
        16 * reduction_input_elements * fidelity_factor
    cycle_denominator = 4096 * core_count
    ideal_fpu_cycles = ceil_div(cycle_numerator, cycle_denominator)
    ideal_fpu_ns = ceil_div(ideal_fpu_cycles * 1000, clock_mhz)

Both implementations must form these products with checked wide integers.
Formula tests include values above 2^53 and exact-boundary cases; no
floating-point accumulation is used for the C++/Python golden formulas.

Never describe 4096 as multiply-adds. It is 4096 FLOPs, or 2048 MACs.

Primary capability references:

- tech_reports/matrix_engine/matrix_engine.md in tt-metal
- https://github.com/tenstorrent/tt-isa-documentation/blob/main/WormholeB0/TensixTile/TensixCoprocessor/MatrixUnit.md

The tt-metal matrix-engine report explicitly applies its matrix-unit throughput
figures to Wormhole and Blackhole.

### 3.3 Explicit SFPU exclusion

The public documentation does not establish a capability for SFPU-only
functions such as exp, sigmoid, SiLU, and reciprocal square root. The first
version therefore excludes those results from ideal_fpu_cycles.

It must:

- not infer SFPU throughput from LLK code or measurements
- not decompose SFPU functions into invented primitive FLOP counts
- expose omitted_sfpu_results so the limitation remains visible
- call the metric fpu_utilization, not total compute utilization

Ordinary mathematical add/subtract/multiply/reduction work explicitly present
in the canonical oracle is still counted, regardless of which current
implementation path executes it. Unary sign change has no documented work
class and is excluded. A named nonlinear primitive is atomic for
this rule: all of sigmoid, SiLU, exp, and reciprocal square root is omitted
rather than inventing an internal primitive decomposition. This is why the
explicit sigmoid-gate multiplication in RMS norm is counted while the
definitional x-times-sigmoid-x inside PyTorch SiLU is not.
Structural masking/index selection such as tril is also excluded because it has
no documented FPU work class.

### 3.4 Aggregate DRAM capability

Use one aggregate bidirectional DRAM resource:

    512 GB/s = 512,000,000,000 bytes/s = 512 bytes/ns

Read and write bytes are summed; they are not treated as independently
overlapped channels:

    mandatory_dram_bytes =
        sum(unique DRAM input physical bytes) +
        sum(unique DRAM output physical bytes)
    ideal_dram_ns = ceil(mandatory_dram_bytes / 512)

Rules:

- count a unique input buffer once even if it appears more than once
- count a unique output buffer once
- identify a DRAM buffer by buffer_address within this single-device model
- if input and output alias, count its required read and required write
- skip an absent optional tensor entirely
- L1 tensors contribute zero
- use physical volume times element size, including padding
- sum all tensors; never use the largest tensor as a proxy
- use decimal SI GB/s, not GiB/s

Supporting reference:

- https://github.com/tenstorrent/tt-low-level-documentation/blob/main/data_movement_doc/general/ideal_performance.md

That document provides ideal Blackhole directed DRAM bandwidth measurements.
The repository's existing Blackhole capability constant/comment supplies the
512 GB/s aggregate value. Decimal SI conversion intentionally differs from
OpPerformanceModelGeneral, which currently converts the value as binary GiB/s;
the KDA model must not reuse that conversion. Reporting zero bytes for L1
tensor slots also intentionally differs from the generic model because L1 is
not mandatory DRAM traffic. Summing all mandatory bytes also intentionally
differs from the generic model's largest-tensor time.

## 4. Reported values

Tracy receives:

- ideal_fpu_ns
- ideal_dram_ns
- ideal_ns
- input and output per-tensor bandwidth arrays

Given measured_ns:

    fpu_utilization_pct = 100 * ideal_fpu_ns / measured_ns
    dram_utilization_pct = 100 * ideal_dram_ns / measured_ns
    roofline_utilization_pct = 100 * ideal_ns / measured_ns

The pure KDA estimate contains:

- ideal_fpu_cycles
- ideal_fpu_ns
- mandatory_dram_bytes
- ideal_dram_ns
- ideal_ns
- omitted_sfpu_results
- input and output per-tensor byte arrays

The profiler hook converts it to
OpPerformanceModelGeneral<tensor_return_value_t>. Tracy therefore receives its
existing compute, bandwidth, ideal-time, and per-tensor bandwidth fields; the
extra KDA values remain available to C++ formula tests and the Python mirror.

Per-tensor Tracy bandwidth arrays have one entry for every present tensor slot
in declaration order, without deduplication, matching the generic profiler
contract; absent optional tensors are skipped. Each entry is physical_bytes /
ideal_ns in GB/s because one byte/ns equals one GB/s. L1 tensors report zero
bandwidth. Buffer deduplication applies only to mandatory_dram_bytes. Zero ideal
time in the pure estimate converts to a generic wrapper with ideal_ns = 1 and
one zero entry per present tensor slot, preventing division by zero without
misreporting traffic.

## 5. Mathematical work totals

All dimensions below are logical operation dimensions. Checked wide-integer
arithmetic is required before conversion to profiler types.

### 5.1 Sigmoid gated RMS norm

Let R = B times S times H be the number of rows and V the row width.

The oracle computes square, mean, epsilon addition, reciprocal square root,
normalization, weight multiplication, sigmoid, and gate multiplication.

    multiply_results = 4 R V
    reduction_input_elements = R V
    add_results = R
    dense_flops = 0
    omitted_sfpu_results = R + R V

The omissions are R reciprocal-square-root results and R V sigmoid results.

### 5.2 QKV causal conv1d with SiLU

Let E = B times S times (Dq + Dk + Dv), the total number of output elements.
Each causal depthwise output applies four taps.

    multiply_results = 4 E
    add_results = 3 E
    reduction_input_elements = 0
    dense_flops = 0
    omitted_sfpu_results = E

The omission is one SiLU result per output element.
Read B from the input logical shape; it is not an operation attribute.

### 5.3 Reduce affine transforms

Let G = groups_per_head, H = batch_heads, K the square state dimension, V the
value width, and P = H times (G - 1).

One affine composition performs A-right times A-left and A-right times B-left,
then adds B-right:

    dense_flops = P times (2 K cubed + 2 K squared V)
    add_results = P K V
    multiply_results = 0
    reduction_input_elements = 0
    omitted_sfpu_results = 0

For G = 1 all work totals are zero.

The initial identity/zero seed simplifies exactly to the first input affine
pair, so it introduces no mathematical operator work.

### 5.4 Affine exclusive scan

The mathematical output contains the initial state and G - 1 state
transitions. It is not modeled as the current parallel scan tree.

Let P = H times (G - 1):

    dense_flops = P times 2 K squared V
    add_results = P K V
    multiply_results = 0
    reduction_input_elements = 0
    omitted_sfpu_results = 0

For G = 1 all work totals are zero.

The final carry computed by a literal loop is unused by the exclusive output
and is therefore not mathematical work of the operation.

### 5.5 Prepare chunk recurrence

Let H be the number of heads, N the number of chunks,
C = tt::constants::TILE_HEIGHT = 32 the chunk length,
K the key width, and V the value width. The following per-head, per-chunk totals
come directly from the oracle, with mathematical common subexpressions counted
once:

    multiply_results = 10 C K + C V
    add_results = 2 C + (C - 1) K + C K + C squared
    reduction_input_elements = 2 C K
    dense_flops =
        4 C squared K +
        C (C - 1) (C + 1) / 3
    omitted_sfpu_results = 2 C + 3 C K + K

Multiply every total by H N.

The final dense term is the exact multiply-plus-add work of the unit-lower
triangular inverse under forward substitution: twice choose(C + 1, 3).
Its integer expression divides exactly because C - 1, C, and C + 1 are three
consecutive integers.
The omitted SFPU results are two C-wide normalization reciprocal square roots,
three C-by-K exponential tensors, and one K-wide exponential tensor.
The unary negation used to form one exponential argument is excluded as stated
in section 3.3. The C-squared addition of identity and a strict-lower tensor is
counted because it is an explicit binary tensor operator; disjoint-support
values do not specialize the structural operator count.

### 5.6 Recurrent chunk scan

Let H = batch_heads, N = num_chunks, C the chunk length, K the state width, and
V the value width.

C is tt::constants::TILE_HEIGHT, not a program-factory value.

Per head and chunk:

    dense_flops = 6 C K V + 4 C squared V
    multiply_results = K V
    add_results = 2 C V + K V
    reduction_input_elements = 0
    omitted_sfpu_results = 0

Multiply totals by H N.

### 5.7 Recurrent chunk scan summary

The summary performs two mathematical state scans plus its final A
subtraction. K = V is a precondition of the operation.

    dense_flops = H N times (8 C K V + 4 C squared V)
    multiply_results = H N times 2 K V
    add_results = H N times (2 C V + 2 K V) + H K V
    reduction_input_elements = 0
    omitted_sfpu_results = 0

These closed totals intentionally count two complete scan evaluations from the
zero and identity states. Those evaluations define the summary API's affine
outputs; the model does not introduce a separate symbolic recurrence
simplifier. Similarly, recurrent mode uses its supplied initial state and has
no zero-state specialization.

## 6. C++ implementation

Add a KDA-local common component:

- ttnn/cpp/ttnn/operations/experimental/kda/kda_performance_model.hpp
- ttnn/cpp/ttnn/operations/experimental/kda/kda_performance_model.cpp

It owns:

- a small KdaWork value with the four documented work classes and SFPU omission
- a pure KdaEstimate with cycles, times, bytes, omissions, and tensor-byte arrays
- checked dimension/work arithmetic
- fidelity-to-factor conversion
- six operation estimators, with recurrent and summary modes represented
- post-harvest core-grid and clock conversion
- mandatory DRAM accounting and buffer deduplication
- conversion from KdaEstimate to the repository's concrete
  OpPerformanceModelGeneral<tensor_return_value_t>

Do not change OpPerformanceModelGeneral. The profiler hook is structurally
detected, but MeshDeviceOperationAdapter fixes its concrete return type; each
KDA hook must therefore return OpPerformanceModelGeneral<tensor_return_value_t>.
Create its default instance and explicitly populate public fields rather than
calling the generic constructor, whose DRAM semantics are intentionally
different.

Each of the six KDA device-operation pairs gains a create_op_performance_model
hook. The hook may gather:

- logical mathematical dimensions
- math fidelity
- device grid size and clock
- input/output tensors for DRAM accounting

It must not call factories or scheduling helpers.

The hook has the repository-standard signature:

    static tt::tt_metal::operation::OpPerformanceModelGeneral<
        tensor_return_value_t> create_op_performance_model(
        const operation_attributes_t&,
        const tensor_args_t&,
        tensor_return_value_t&)

There is no device parameter. Obtain the device from a present device-resident
input tensor after checking StorageType::DEVICE. An unexpected non-device case
uses the warning/zero-estimate fallback from section 3.1. Populate
ideal_compute_cycles from ideal_fpu_cycles, ideal_compute_ns from ideal_fpu_ns,
ideal_bandwidth_ns from ideal_dram_ns, ideal_ns from ideal_ns, and the byte
arrays from the KDA estimate. Checked narrowing to the generic int fields is
required. Clamp the wrapper's cycle and time fields to at least 1; estimator
and Python values remain zero where mathematically appropriate.

Pure estimators take core_count and integer clock_mhz as values. Device
discovery stays
outside them so harvested/non-square cases are deterministic unit tests.

## 7. Python realtime-profiler mirror

Add:

- tests/ttnn/nightly/unit_tests/operations/experimental/kda/kda_performance_model_test_utils.py
- tests/ttnn/nightly/unit_tests/operations/experimental/kda/test_kda_performance_model.py

The Python model mirrors the C++ formulas and constants exactly. Extend the
realtime-profiler utility to expose frequency_ghz from the runtime data it
already reads. This frequency is a regression estimate derived from device
cycles and host timestamps; it is not the same measurement as C++ ARC
get_clock_rate_mhz. Python performance logs use the realtime frequency because
it is paired with measured runtime, converted once to integer MHz as
floor(frequency_ghz times 1000 + 0.5). C++ Tracy estimates use integer ARC
AICLK. Both then use the exact ceil-div clock conversion from section 3.2.

Update all nine KDA performance-test entry points across the seven operation
test files to log:

- measured_ns and runtime_id
- ideal_fpu_cycles and ideal_fpu_ns
- mandatory_dram_bytes and ideal_dram_ns
- ideal_ns and omitted_sfpu_results
- fpu_utilization_pct, dram_utilization_pct, and roofline_utilization_pct

No kernel or operation implementation change is required to produce these
metrics.

## 8. Tests and validation

### 8.1 Formula tests

Add focused C++ gtests and register them in UNIT_TESTS_TTNN_BASIC_SOURCES. Add
matching Python golden tests.

Cover:

- every operation and recurrent/summary mode
- minimal and production-sized dimensions
- G = 1 zero-work scan/reduction cases
- all four fidelity factors
- harvested and non-square device grids
- exact omitted-SFPU counts
- DRAM-only, L1-only, and mixed placement
- multiple inputs/outputs proving sum rather than max
- repeated-buffer deduplication and input/output alias semantics
- physical padded volume
- decimal 512-byte/ns rounding
- overflow, unsupported-architecture, invalid-fidelity, and non-device-tensor
  fallback behavior

The C++ and Python expected values must be independently calculated from the
closed formulas, not copied from one implementation to the other.

### 8.2 Tracy validation

For one exact pytest item per KDA API, run:

    python -m tracy -p -r -v -m pytest <exact-test-item>

Compare Tracy's compute, bandwidth, and ideal nanoseconds, plus per-tensor
bandwidth arrays against the independently calculated expectations available
from Tracy. DRAM time and byte-derived bandwidths must match exactly. Do not
directly equate Tracy compute_ns with the realtime Python compute_ns because
their clock estimators differ. Instead, validate exact work cycles and
clock-to-time conversion in C++ and Python formula tests with an injected
clock, and sanity-check both hardware values against their own recorded clock
where available. Cycles, byte totals, omitted-SFPU counts, and utilization
percentages are validated in Python goldens because they are not all present in
Tracy's result schema.

### 8.3 Hardware validation

Run every parametrized item produced by the nine KDA performance entry points
on Blackhole through the workspace's safe hardware-test workflow. Report exact
commands and results, including
pass/fail, PCC where applicable, measured timing, modeled FPU/DRAM/roofline
times, and all three utilization percentages.

## 9. File-impact boundary

Expected changes are limited to:

- the new KDA common C++ performance-model files
- ttnn/cpp/ttnn/operations/experimental/kda/CMakeLists.txt to compile the new
  common source
- the six KDA device-operation header/source pairs for profiler hooks
- the new KDA Python model and its formula tests
- the C++ gtest and source registration
- the realtime-profiler utility and its focused tests
- the seven existing KDA Python performance-test files

Explicitly unchanged:

- ttnn/core/operation.cpp and the generic model
- KDA factories, program grids, kernels, and LLK code
- other operations' performance models
- frozen functional and accuracy gates

## 10. Implementation sequence

1. Add independently checked C++ and Python work/formula tests.
2. Implement the KDA common C++ model and KDA-local DRAM accounting.
3. Attach the six operation hooks without consulting scheduling code.
4. Implement the Python mirror and frequency plumbing.
5. Add the standard fields to all nine performance-test entry points.
6. Run unit, Tracy, and Blackhole validation.
7. Commit one validated concern at a time.

## 11. Risks and explicit limitations

- SFPU work is deliberately omitted, so fpu_utilization is a partial documented
  FPU roofline, not total compute utilization.
- The triangular inverse work is charged at peak dense-matrix FPU throughput,
  an optimistic lower-bound assumption.
- Perfect work balance, perfect overlap, and use of every harvested core make
  the result a lower bound rather than a runtime forecast.
- The 512 GB/s value is an aggregate Blackhole assumption. Supporting docs and
  the repo constant must be cited beside the implementation constant.
- Mathematical-oracle changes require updating both implementations and both
  independent golden suites.
- Unsupported architectures return a visible zero model rather than a
  misleading Blackhole estimate.

## 12. Approval

Approved by the user on 2026-08-28 for autonomous local implementation.
Commits and review artifacts remain local; do not push or open a pull request.
