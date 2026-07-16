# Which tracing is this about?

There are three systems in the Tenstorrent ecosystem that may be called tracing:

1. Metal tracing, also known as trace capture/replay, TTNN trace, device trace, captured trace, trace replay, command/dispatch trace.

```
# Begin recording operations
tid = ttnn.begin_trace_capture(device, cq_id=0)
output = run_model(input)
ttnn.end_trace_capture(device, tid, cq_id=0)

# Replay the traced operations
ttnn.execute_trace(device, tid, cq_id=0)
```

2. TT-NN Graph Tracing, also known as graph capture, graph trace, C++ graph capture, runtime graph capture, captured graph, graph report.

```
ttnn.graph.begin_graph_capture()
```

3. TT-NN Tracer, also known as computation-graph tracing, operation tracing, model tracing, Torch/TTNN tracing, NetworkX graph tracing

```
ttnn.tracer.trace()
```

This guide is about system 1 - trace capture/replay.

# What does tracing do?

Metal Trace is a performance optimization feature that minimizes host overhead for constructing and dispatching operations. It is especially useful when the time to execute the operation on device is shorter thatn the time needed to dispatch it, which is commonly the case with small tensors and simple operations.

# How does it do it?

When capturing, the operation dispatching commands are saved into a DRAM buffer allocated for this purpose.
The commands contain the addresses of the operands, input/output shapes etc.

When replaying, the same chain of operations is executed again. Whichever addresses were read from and written to during capture will be read from and written to again. No allocator operations or checks are performed. Importantly, the operation's program factory (#TODO link to explanation of what this is) doesn't run again.

# Requirements and important details

## Full pre-compilation

Before capturing a trace, all programs must already be compiled and present in the program cache.
This means the exact path needs to be executed before starting capture.
Some operations perform auto-tuning and may compile differently (with different program cache keys) based on parameters other than operation arguments, for example available L1 memory. For a detailed example, see https://github.com/tenstorrent/tt-metal/issues/46533.
Make sure to replicate the conditions faithfully in warmup. If a program cache miss happens during trace capture, an error will be printed. 

## Unsafe allocations

The captured chain of operations may include allocations and deallocations of temporary tensors. After capture, these will no longer be visible to the allocator. If you allocate a new buffer after capturing a trace, it may include addresses that will be overwritten when that trace gets executed again.
As a consequence, all allocations made after some trace has been captured, and not released before it is executed, are unsafe.

You can use the trace allocation tracker tool to check a program for potentially unsafe allocations.

If we capture multiple traces, we may want to persist the input tensors as placeholders where new inputs will be copied, while being aware that the old contents may be corrupted by executing other traces. To exempt such tensors from validation use mark_corruptible(tensor).

Buffer allocations may not be explicit!
For example `ttnn.reshape`, when first executed with a new shape, may allocate a lookup table to speed up future executions. If this happens after some trace has already been captured, executing that trace may corrupt this table. This will lead to undefined behavior, including data corruption and hangs.
When a new program is compiled, the compilation result is saved in program cache. Program cache buffers share the address space with regular buffers - they may also be corrupted, if not allocated before all trace capture.

## Unexpected compilation parameters

Operation arguments may be compile time constants even if you don't expect them. For example, with integer `k` in `ttnn.slice(tensor, k)`, different `k` values will lead to recompilation. When tracing, look for variants where changing parameters are runtime arguments. For example, `ttnn.slice` has a variant which accepts the slicing index as a tensor. When traced, trace execution will read new index from the captured tensor and not recompile when it changes.

## Key the cache correctly 

Be careful when tracing bigger operation blocks, such as a full model forward pass.
When the trace is replayed, the python code is not executed again. If there are conditional statements deciding which device operations get executed, they will not be re-evaluated on trace execution.
You can record separate traces for different paths, and replay the right one depending on the conditions.

If there are multiple decision points and capturing traces for all combinations is not feasible, consider splitting the traced operation into sections.

## Dynamic trace buffer allocation

Usually, the traces are recorded into a pre-allocated trace region, with its size set when opening the device.
This means you need to appropriately size the buffer to accomodate all traces you plan to capture.

You can avoid this by passing `trace_region_size=0`. With this set, traces will be saved in dynamically allocated buffers.
As of July 2026, dynamic trace buffer allocation should be avoided when using multiple traces due to a bug https://github.com/tenstorrent/tt-metal/issues/48869

# Common pattern

```
class FunGen:
    def fun(x, cond_a, cond_b):
        if cond_a:
            return op_a1(x)
        else:
            return op_a2(x)
        if cond_b:
            return op_b1(x)
        else:
            return op_b2(x)

    def __init__(self):
        self.trace_inputs = dict()
        self.trace_ids = dict()
        self.trace_outputs = dict()

        # warmup - compilation
        for cond_a in True, False:
            for cond_b in True, False:
                x = create_x()
                fun(x, cond_a, cond_b)
        # trace capture
        for cond_a in True, False:
            for cond_b in True, False:
                trace_key= (cond_a, cond_b)
                x = create_x()
                trace_id = ttnn.begin_trace_capture(self.mesh_device, cq_id=0)
                out = fun(x, cond_a, cond_b)
                ttnn.end_trace_capture(self.mesh_device, trace_id, cq_id=0)
                # These two may be overwritten by executing other traces,
                # But we always write them before use
                # and read from them right after executing their trace
                mark_corruptible(x)
                mark_corruptible(out)
                self.trace_inputs[trace_key] = x
                self.trace_ids[trace_key] = trace_id
                self.trace_outputs[trace_key] = out

    def traced_fun(x, cond_a, cond_b)
        trace_key = (cond_a, cond_b)
        trace_input = self.trace_input[trace_key]
        trace_output = self.trace_output[trace_key]
        trace_id = self.trace_ids[trace_key]
        copy_data(from=x, to=trace_input)
        ttnn.execute_trace(device, trace_id, cq_id=0, blocking=False)
        # Need to move the output to non-corruptible memory before any other trace is executed
        # This can be a host copy, or a copy to some other safe tensor.
        result = trace_output.cpu()
        return result
