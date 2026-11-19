# Full-attention norm output stays in projection-compatible L1

Source-only candidate based on frozen `candidates/integrated134.py`. No device
execution, numerical equivalence or performance gain is claimed yet.

Parent profile evidence for final118 full attention identifies, per hidden
norm, approximately 3 us BF16-to-FP32 interleaved-to-sharded, 9 us native
eight-core RMSNorm, 3 us FP32-to-BF16 sharded-to-interleaved, then 2 us
interleaved-to-sharded for projection input. The proposed change targets the
last two operations. Those approximate profile durations are motivation, not
a forecast for saved whole-decoder latency.

The norm produces FP32 WIDTH_SHARDED L1 on eight row-major cores with shard
shape `[32,640]`. Full-attention default packed_qkv/gate and MLP gate/up have
the identical projection input MemoryConfig. Instead of narrowing into DRAM
and re-sharding, the candidate performs:

```python
ttnn.typecast(y, ttnn.bfloat16, memory_config=memory)
```

Here `y` is the **unchanged native FP32 RMSNorm result**. The following
projection's `to_memory_config(x, info["input_mem"])` should return the same
buffer when configuration already matches. Existing `share_mlp_input` shares
that buffer across gate and up. The candidate does not change residual
residency, projection output residency, norm input precision, gamma,
epsilon, compute config, native reduction topology or projection geometry.

This is deliberately distinct from the previously slower whole-L1 residual
policy. Only one 320 KiB logical BF16 normalized activation (40 KiB/core,
including tile padding) persists in L1; native norm FP32 temporaries retain
their existing lifetimes. Allocation coexistence must still be measured in the
whole decoder, particularly when the activation remains live across MLP down.

## Source/API boundaries

`ttnn/cpp/ttnn/operations/normalization/rmsnorm/rmsnorm_nanobind.cpp:107-120`
does not expose an output-dtype argument. The C++ wrapper passes `std::nullopt`
as primitive dtype (`rmsnorm.cpp:82`), so a hypothetical direct BF16-output
RMSNorm call is not used. Changing norm input to BF16 would change its precision
path and is outside this candidate. The existing `_norm` implementation already
uses `ttnn.typecast(y, BF16)` for the `l1_mlp` post-norm branch, establishing the
API route independently of this experiment.

The new branch applies only to full attention and only when all selected
projection consumers have exactly the norm output MemoryConfig. It checks
packed or separate attention consumers and both separate MLP projections.
Unmatched geometry follows the existing DRAM publication. The flag
`norm_projection_l1=False` disables the candidate for controlled comparison.
The GDN norm/runtime is untouched.

Source placement:

- Norm branch and eight-core shape: `candidates/integrated134.py:526-613`.
- Projection memory creation: same file around 350-365.
- Projection conversion: `_linear_raw` around 446-450.
- MLP shared input: `_finish` around 666-682.
- Compact attention consumes the normalized tensor through packed_qkv and
  gate: `_compact_attention_decode` around 1584-1602 and 1669.

## Immutable artifacts and checks

- `candidates/norm_l1_handoff_on134.py`: frozen candidate.
- `candidates/make_norm_l1_handoff_candidate.py`: generator, refuses overwrite.
- `tests/run_norm_l1_handoff_probe.py`: isolated device probe; **not run**.

Generator execution and `python -m py_compile` passed. The change adds one
guarded branch within `_norm`; all existing arithmetic source and other
methods remain unchanged. Source agreement does not prove the two FP32-to-BF16
conversion implementations round identically; that is the isolated probe's
primary purpose.

The probe loads real checkpoint folded gamma (`1 + HF weight`) for both hidden
norms of full-attention layer 3. At B1/3/32 and four activation scales, it
computes native FP32 norm **once**, then compares old DRAM-narrow-and-reshard
against the new L1 typecast. It requires bit equality, identical memory config,
and buffer-address aliasing through the next projection conversion. Torch
readback is diagnostic only, outside any measured production path.

Recommended command through the parent's serialized device queue:

```bash
python_env/bin/python -m models.autoports.qwen_qwen3_6_27b.tests.run_norm_l1_handoff_probe \
  --layer 3 --batch 1 3 32 \
  --output models/autoports/qwen_qwen3_6_27b/doc/optimized_decoder/norm_l1_handoff_probe.json
```

Then select `norm_l1_handoff_on134.py` using the normal evidence decoder
class/kwargs override, run full-attention correctness and repeated traced
B1/B32 latency, and inspect profiler rows for removed transfers. Keep only a
whole-path improvement that preserves the existing gates. If rounding differs,
record the exact mismatch and investigate its source before considering it a
correctness-preserving candidate; do not infer correctness from fewer ops.
