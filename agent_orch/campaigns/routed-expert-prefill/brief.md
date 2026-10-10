# routed-expert-prefill

## What the code does
`ttnn.experimental.deepseek_prefill.unified_routed_expert_moe` runs the routed experts of one prefill MoE layer
on one chip. Input `x` is the dispatch buffer: ROW_MAJOR bf16 `(dispatch_buffer_rows, emb)`, where each local
expert owns a tile-aligned region starting at `offsets[global_id]` with `counts[global_id]` valid rows
(`global_expert_idx_table` maps local -> global expert ids). For each local expert the op computes
`out = act(x @ W_gate [+ b_gate], x @ W_up [+ b_up]) @ W_down [+ b_down]` on that expert's rows and writes the
result at the same offset in the output. Counts and offsets live on device, so the kernels read them at run
time and skip empty experts; the chunk size per expert is picked at run time from the token count
(`kernels/adaptive_chunk.hpp`). Weights are bf4 (`bfloat4_b`), DRAM ND-sharded; the compute config is LoFi,
bf16 dest, packer L1 accumulation. The activation (Silu / SituGlu / SwiGluOai / ClampedSiluGlu) and the
optional biases are compile-time defines.

The op always runs on a fixed 11x8 core grid (`kCoreGridX/Y` in `device/unified_routed_expert_ffn_types.hpp`).
M is split across grid rows, N across grid columns; the gate+up matmuls share each x K-block, x is tilized
in the compute kernel (bf16 -> bf8_b), and the activated intermediate is multicast in L1 as the K-blocks of
the down matmul.

## Where the code is
Everything editable is under `ttnn/cpp/ttnn/operations/experimental/deepseek_prefill/unified_routed_expert_ffn/`:
- `device/unified_routed_expert_ffn_program_factory.cpp`: grid, block/subblock sizes, circular buffers,
  compile-time and runtime args (host side; changing it needs a rebuild)
- `device/kernels/dataflow/unified_routed_expert_ffn_reader.cpp`: reads counts/offsets, x sticks, weight
  shards (`kernels/weight_runs.hpp` coalesces ND-shard rows into single NoC reads), biases; L1 multicast
- `device/kernels/dataflow/unified_routed_expert_ffn_writer.cpp`: down-projection weights and output writes
- `device/kernels/compute/fused_swiglu.cpp`: tilize, gate/up matmul, activation, down matmul
- `device/kernels/adaptive_chunk.hpp`: run-time chunk/per_core_M choice shared by all three kernels
- the device-op, types, op wrapper and nanobind files
Kernels are JIT-compiled at run time; host files need a rebuild.

The Blackhole LLK compute APIs (`tt_metal/hw/ckernels/blackhole/`: matmul, tilize, pack/unpack, SFPU) are
also editable. They are shared by every op, so keep changes behind defines or new functions that only this
op's kernels use, and keep the existing APIs' behavior unchanged.

## What is measured
`tests/ttnn/nightly/unit_tests/operations/fused/test_routed_expert_prefill.py`: 9 cases, one per production
model (DeepSeek-V3, Kimi-K2.7, Kimi-K3 latent, GLM-5.3, Mistral-Small-4, DeepSeek-V4 Flash/Pro, MiniMax-M3,
GPT-OSS-120B), each the local shapes of one chip of that model's production prefill mesh (experts per chip,
dispatch buffer rows, uniform top-k token counts). Emb 2880-7168, intermediate 2048-3072, all four
activations, and biases for GPT-OSS. 3 warmup + 10 measured calls per case. The value per case is the mean
DEVICE KERNEL DURATION of the measured calls, in µs, from the Tracy ops CSV. Accuracy gate: PCC >= 0.979 on
every case (max_abs is also recorded).

The op is mostly bound by streaming bf4 weights from DRAM and by matmul throughput at LoFi. Cases differ
widely in tokens per expert, so a change that helps one shape can hurt another: check all 9.

## Constraints
See the campaign rules: fidelity, approximation and intermediate precision stay at the baseline, the
input/weight/output formats are fixed by the test, and every activation, the bias path and the hybrid
token band (`min/max_active_tokens`, used by `hybrid_routed_expert_ffn`) must keep working.
