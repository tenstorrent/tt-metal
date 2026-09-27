# Final-policy BFP4 expert packing audit

The sliding-layer integration changed how indexed expert gate/up weights become
BFP4. Restoring the selected candidate's direct host packing is sufficient to
restore its complete PCC sequence. The imported source weights are BF16; there
is no intermediate BFP8 weight quantization in the integrated conversion path.

This is a CPU/source-only follow-up. No TTNN import, device access, native change,
or runtime change was performed by this audit. Hardware results below are the
root agent's saved artifacts, independently compared using Python's JSON reader.

## Controlled evidence

All rows use the sliding layer, real weights, prefill length 4096 and 128 decode
positions. The latter two use eight duplicate replays per position.

| Artifact | Runtime prefix | Expert gate/up construction | Minimum decode PCC |
| --- | --- | --- | --- |
| `sliding_combined_shared_gate_bfp4_geom2.json` | `c74f191d` | CLI direct host BFP4 packing | 0.9976112404200184 |
| `sliding_final_policy_stress.json` | `1683b8a4` | Device BF16-to-BFP4 typecast | 0.9912606698254975 |
| `sliding_final_raw_gate_control.json` | `1683b8a4` | Same integrated runtime, only `--expert-gate-bfp4` added | 0.9976112404200184 |

The candidate and raw-packing control have exactly equal **all 129 PCC entries**,
including prefill. All three prefill PCCs are 0.9999220421496003, and all three
cache-PCC arrays are exactly equal. Both integrated runs retain exact replica and
duplicate-replay checks. The failed/control JSON objects differ only in
`expert_gate_bfp4`, command, PCC, pass status, and timings. This is a deterministic
numerical regression; the earlier router-placement nondeterminism is a separate
issue.

The isolated control establishes the sufficiency of reproducing weight
construction. It does not, by itself, identify the exact hardware rounding rule
or constitute a bytewise comparison of the two packed weight buffers.

## Source path and packing order

1. `tests/run_decoder.py:load_layer` loads the checkpoint tensors and calls
   `.float()`. The existing functional-stage `weight_stats.json` records layer-0
   `experts.gate_up_proj` as BF16, shape `[128, 1408, 2816]`. Thus the host floats
   preserve BF16-valued source weights exactly.
2. `tt/multichip_decoder.py` constructs `Gemma4DecoderLayer` with `dtype=BF16`,
   no `experts_dtype` override, and no tensor cache path. The defaults in
   `models/demos/gemma4/tt/layer.py`, followed through `tt/moe.py`,
   `tt/experts/__init__.py`, and `tt/experts/weights.py`, pass BF16 to the expert
   tensor uploads. The weight loader's standalone BF8 default is overridden.
3. The native loader splits contiguous gate/up halves, transposes each to
   `[1, 128, 2816, 704]`, pads 64 columns globally to 768, and shards the last
   dimension across TP4. Each rank receives 192 columns per projection.
   `PackedExperts` in `tt/fused_decoder.py` concatenates its BF16 gate and up
   tensors locally, giving `[1, 128, 2816, 384]` per rank.
4. `OptimizedExperts.__init__` in `tt/optimized_decoder.py:104` applies
   `ttnn.typecast(source.gate_up, gate_dtype)` to that BF16 device tensor.
   Setting `gate_dtype=BFP4` therefore chooses device BF16-to-BFP4 conversion.
5. The candidate branch in `tests/run_multichip_decoder.py` instead splits and
   transposes the original host tensor, applies the same global padding, splits
   gate and up into four chunks, then concatenates
   `[gate_rank0, up_rank0, gate_rank1, up_rank1, ...]`. It uploads this directly
   with BFP4 dtype and a last-dimension mesh mapper.

Consequently the logical weights, gate/up order, expert order, padding location,
TP ownership, and tile boundaries agree before quantization. The per-rank width
192 is tile aligned; gate/up concatenation introduces no shared-exponent group
crossing a projection boundary. Padding is global before splitting, not 16 new
zeros appended separately to each unpadded 176-column rank slice.

## Conversion implementation distinction

The host path dispatches to `pack_as_bfp4_tiles` in
`tt_metal/impl/data_format/bfloat4.cpp`, then to
`pack_as_bfp_tiles<Bfp4_b>` in `blockfloat_common.cpp`. The latter selects a shared
exponent for each 16-element face row. Its
`convert_u32_to_bfp<BfpFormat, false>` call explicitly rounds mantissas to nearest,
ties to even, and clamps to the representable mantissa range.

The device path is different:

- `ttnn/cpp/ttnn/operations/copy/typecast/typecast.cpp` selects
  `fp32_dest_acc_en=false` for BF16-to-BFP4. Its `bfp8_pack_precise` flag is true
  only when the output is BF8, so this BFP4 path selects approximate BFP packing
  in `device/typecast_program_factory.cpp:make_compute_config`.
- `device/kernels/compute/eltwise_typecast.cpp` copies into destination registers
  and calls `pack_tile` for the output format.
- `tt_metal/hw/inc/api/compute/eltwise_unary/typecast.h:339` explicitly implements
  Float16_b-to-Bfp4_b as no SFPU conversion: the packer performs it.

These are separate numerical conversion implementations. Host nearest-even
rounding is source-proven; the exact hardware BFP4 tie/saturation behavior is not
established by this audit. No native fix is needed to reproduce the selected
model policy, and no claim is made that changing the BF8 precision flag would
provide an equivalent BFP4 fix.

## Other candidate settings checked

The saved pre-integration runtime (`runtime_before_final_policy.py.txt`), current
constructor, candidate CLI implementation, and artifacts agree on the following
effective sliding policy:

| Setting | Candidate and integrated effective value |
| --- | --- |
| Decode QKV | LoFi, grid 8x4, K block 22, per-core N and output subblock width 2 |
| Decode WO | LoFi, grid 11x8, K block 32 |
| RoPE and attention CCL | BF16 sharded decode RoPE; BF16 attention CCL |
| Expert decode | BF8 input; BFP4 gate/up and down; LoFi; baseline sparse geometry |
| Shared decode | Raw host BFP4 gate/up, BF8 down, geometry 2 |
| MoE reduction | Grouped BF8 CCL |
| Prefill and tail | Hybrid EP prefill, existing BF16 shared prefill, fused tail |

The CLI QKV override supplies `out_block_h=1, out_block_w=2` explicitly. The
constructor omits them, but the nanobind constructor in
`ttnn/cpp/ttnn/operations/matmul/matmul_nanobind.cpp:367` defaults them to
`per_core_M=1, per_core_N=2`; this is not a mismatch. The full-attention CCL
default does not affect this sliding layer. The CLI booleans recorded as false
in integrated artifacts mean no override was requested, not that the new
constructor policy was disabled.

## Integration boundary

The root agent has integrated the candidate's raw host BFP4 construction for
sliding indexed experts only. The current source preserves full-layer device
conversion. In hybrid mode it aliases the unused indexed-prefill gate pointer
to the replacement decode weight; `_HybridExperts` delegates actual prefill to
the independent EP object. This avoids retaining the superseded BFP4 decode
allocation through that alias and preserves the separate prefill policy.

The raw-only control is already conclusive for the reported PCC regression.
The ordinary default stress run of the integrated source remains the root
agent's final validation. Nonhybrid alternatives, other layer kinds, and native
host/device BFP4 equivalence are outside that isolated control's evidence.

Checks performed here: source inspection, saved-runtime diff, aggregate JSON
comparison, and report whitespace check. This is a documentation-only change;
no build or hardware test was run by this agent.
