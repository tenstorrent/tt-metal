# Host Reduce Helper for tt-metal

## Purpose

Add a host-side reduction-planning helper for tt-metal. The helper must turn one logical reduction request into:

1. a sequence of self-contained compute-kernel reduce-call descriptors; and
2. one auxiliary-tile descriptor for the dataflow kernel(s).

The resulting descriptors are serialized into kernel compile-time arguments. An operation may invoke the planner more than once, producing multiple independent planning units.

This document defines the required behavior and interface boundaries. It intentionally leaves the exact call fields, auxiliary-tile contents, and low-level serialization types to the implementer, because those details depend on the reduce implementation and planner decisions.

## Terminology

### Input tensor description

The planner accepts a vector of input tensor descriptions. Each description contains at least:

- the circular buffer (CB) containing the tensor;
- the tensor shape; and
- the dimension along which that tensor is reduced.

Different input descriptions may refer to the same CB. The interface must not assume one CB per input tensor or that all inputs use distinct CBs.

### Reduce-call descriptor

A reduce-call descriptor is the complete description of one call to the compute-kernel `reduce()` functionality.

### Auxiliary-tile descriptor

An auxiliary-tile descriptor tells the dataflow kernel(s) which auxiliary tiles must be created for one planning unit. Its contents depend on the decisions made while planning the corresponding compute calls.

### Planning unit

One invocation of the planner produces one planning unit. A planning unit contains:

- one ordered sequence of compute reduce-call descriptors; and
- exactly one auxiliary-tile descriptor shared by every reduce call in that sequence.

## Conceptual Host-Side Interface

The concrete C++ names and container types may follow tt-metal conventions, but the logical interface should be equivalent to:

```cpp
struct InputTensorDescription {
    /* CB identifier */ cb;
    /* tensor shape */ shape;
    /* dimension to reduce */ reduction_dim;
};

struct ReduceCallDescription {
    // Every datum required to issue this particular reduce() call.
};

struct AuxiliaryTileDescription {
    // Every datum required to create the auxiliary tiles for this unit.
};

struct ReducePlanningUnit {
    std::vector<ReduceCallDescription> calls;
    AuxiliaryTileDescription auxiliary_tiles;
};

ReducePlanningUnit plan(/* reduction configuration, */
                        std::span<const InputTensorDescription> inputs);
```

The reduction configuration may include information such as the reduction operation when that is not already fixed by the helper's type or context. Its exact representation is an implementation decision.

## Required Planner Semantics

### Reduce all listed inputs together

All input tensor descriptions passed to one `plan()` invocation belong to one logical reduction result. They must be reduced together, rather than producing an independent result for each input.

When realizing that logical result requires more than one compute-kernel `reduce()` call, the calls must use cross-call accumulation so that every applicable call contributes to the same output.

Any state that controls cross-call accumulation—such as whether a call initializes, continues, or finalizes an accumulation—must be represented explicitly in the individual call descriptor when needed. It must not be inferred solely from the call's index in the sequence.

### Produce an ordered sequence of compute calls

One planning unit may contain one or many reduce calls. The planner chooses the number and contents of those calls from the input descriptions and reduction configuration.

The planned order is the execution order. Requiring each descriptor to be self-contained does not imply that calls are freely reorderable; it means that interpreting a descriptor must not require inspecting its neighbors or its position.

### Every compute-call descriptor is self-contained

Each `ReduceCallDescription` must contain all data needed to invoke `reduce()` for that call. Given a descriptor and the normal kernel context, compute-kernel code must not need to derive call behavior from:

- the descriptor's position in the sequence;
- a preceding or following call descriptor; or
- implicit per-sequence state that was omitted from the descriptor.

This includes every call-specific choice made by the planner, including accumulation behavior where applicable. The implementer should derive the exact fields from the final `reduce()` interface.

### Produce exactly one auxiliary-tile descriptor

Every planning unit must contain exactly one `AuxiliaryTileDescription`, regardless of:

- the number of input tensor descriptions;
- the number of CBs used by those inputs; or
- the number of compute reduce calls emitted by the planner.

The auxiliary tiles described by it are planning-unit resources. They are created once at the beginning of the relevant dataflow-kernel work for that unit and shared by all `reduce()` calls belonging to the unit. They must not be recreated separately for every call.

The descriptor must contain all information the dataflow side needs to create the auxiliary tiles selected by the planner. The exact tiles and descriptor fields are deliberately left to the implementation.

## Compile-Time Argument Contract

Planner output is appended to the end of the operation's existing compile-time argument lists. Existing arguments remain a prefix and retain their current meaning.

### Compute-kernel arguments

For each planning unit, append one block with this format:

```text
[num_calls][call_0][call_1] ... [call_(num_calls - 1)]
```

Here, each `[call_i]` is the compile-time-argument serialization of one complete `ReduceCallDescription`.

With multiple planning units, append their blocks consecutively in the same order in which `plan()` was invoked:

```text
[existing compute compile-time arguments]
[unit_0.num_calls][unit_0.call_0] ... [unit_0.call_N]
[unit_1.num_calls][unit_1.call_0] ... [unit_1.call_M]
...
```

Each unit's call count applies only to that unit. Calls from different units must not be merged into a single count or sequence.

### Dataflow-kernel arguments

Append exactly one serialized auxiliary-tile descriptor per planning unit to the end of the applicable dataflow-kernel compile-time arguments. Descriptors use the same planner-invocation order as the compute blocks:

```text
[existing dataflow compile-time arguments]
[unit_0.auxiliary_tiles]
[unit_1.auxiliary_tiles]
...
```

The encoding must let dataflow-kernel code locate and decode each unit's descriptor unambiguously. Whether this is achieved with a fixed-width representation or explicit sizing is an implementation detail.

### Ordering correspondence

Planner invocation order is the common identity between the compute and dataflow argument streams:

```text
first plan() call  -> first compute-call block  + first auxiliary descriptor
second plan() call -> second compute-call block + second auxiliary descriptor
...
```

This ordering must be preserved even when planning units use different reduction operations, input layouts, call counts, or auxiliary-tile requirements.

## Kernel-Side Consumption

### Compute kernel

For each planning unit, the compute kernel must:

1. read that unit's `num_calls`;
2. decode each complete call descriptor in order; and
3. invoke `reduce()` once from each decoded descriptor.

The kernel must be able to invoke `reduce()` using the current descriptor alone, without using the loop index to fill in omitted call semantics.

### Dataflow kernel

For each planning unit, the relevant dataflow kernel must:

1. decode that unit's single auxiliary-tile descriptor at the beginning of the unit's work;
2. create or prepare the described auxiliary tiles once; and
3. make those same tiles available to every reduce call in the planning unit.

No per-call auxiliary descriptor should be emitted or required.

## Multiple Planning Units in One Operation

An operation may invoke the planner multiple times. For example, one operation may plan a sum reduction and then a max reduction.

Each invocation is independent:

- it receives its own vector of input tensor descriptions;
- it produces its own sequence of reduce-call descriptors;
- its calls perform cross-call accumulation only for that unit's logical output;
- it produces its own single auxiliary-tile descriptor; and
- its compute and dataflow metadata are appended after all previously planned units.

For example:

```text
plan(sum_inputs) -> { sum_calls, sum_auxiliary_tiles }
plan(max_inputs) -> { max_calls, max_auxiliary_tiles }

compute suffix:
    [sum_call_count][sum_calls...]
    [max_call_count][max_calls...]

dataflow suffix:
    [sum_auxiliary_tiles]
    [max_auxiliary_tiles]
```

The sum and max units must not share call sequences, call counts, or auxiliary-tile descriptors merely because they belong to the same operation.

## Required Invariants

An implementation must preserve all of the following:

1. One `plan()` invocation corresponds to one planning unit.
2. A planning unit consumes a vector of input descriptions, each containing CB, shape, and reduction dimension.
3. Multiple inputs may refer to the same CB.
4. All inputs within a unit contribute to one combined reduction through cross-call accumulation as needed.
5. A unit produces an ordered sequence of one or more self-contained reduce-call descriptors.
6. Every call descriptor is sufficient on its own to issue its `reduce()` call; descriptor position is not an implicit argument.
7. A unit produces exactly one auxiliary-tile descriptor.
8. The unit's auxiliary tiles are created once at the start of its dataflow work and shared across all of its reduce calls.
9. Each compute argument block is encoded as `[num_calls][calls...]`.
10. Multiple units remain independent and are appended in planner-invocation order.
11. Compute blocks and auxiliary descriptors use the same unit ordering.

## Acceptance Scenarios

The implementation should demonstrate at least the following cases:

### One input, one compute call

- The planner returns one call descriptor and one auxiliary descriptor.
- The compute suffix is `[1][call]`.
- The dataflow side creates the unit's auxiliary tiles once.

### Multiple inputs sharing one CB

- Two or more input descriptions refer to the same CB.
- The planner accepts them without treating the CB as a unique tensor identity.
- All inputs contribute to the same logical reduced output.

### Multiple inputs spanning different CBs

- The planner emits as many calls as its strategy requires.
- Cross-call accumulation combines their contributions into one output.
- Each call carries its own complete CB and accumulation-related information.
- All calls share the unit's one set of auxiliary tiles.

### One input requiring multiple compute calls

- The planner emits more than one call because of shape, dimension, or kernel constraints.
- Every call descriptor remains independently decodable and executable.
- The descriptors explicitly express any required accumulation behavior.
- Only one auxiliary descriptor is emitted for the unit.

### Multiple planning units

- One operation invokes the planner at least twice, such as once for sum and once for max.
- Both compute blocks are appended in invocation order, each with its own call count.
- Both auxiliary descriptors are appended in the corresponding order.
- Neither unit relies on the other unit's calls or auxiliary descriptor.

## Implementation Details Deliberately Left Open

The implementer should resolve these against the existing tt-metal kernel APIs and conventions:

- exact C++ type and function names;
- exact fields and serialized width of `ReduceCallDescription`;
- exact fields and serialized width of `AuxiliaryTileDescription`;
- representation of CB identifiers, shapes, dimensions, and reduction kinds;
- how invalid or mutually incompatible input descriptions are reported;
- how kernel code is told how many planning units to decode;
- whether auxiliary descriptors are fixed-width or explicitly sized;
- how offsets into the appended compile-time-argument suffix are exposed to each kernel; and
- which applicable dataflow kernel(s) consume the auxiliary descriptor in an operation with multiple dataflow kernels.

These choices may vary, but they must not weaken the required semantics or invariants above.
