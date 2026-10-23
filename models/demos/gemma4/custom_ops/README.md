# Gemma 4 custom ops

Model-local fused kernels (`ttnn.generic_op`), the GLM 5.3 `custom_ops/` pattern. The lane decode step of the
26B-A4B on the Blackhole Galaxy runs ~2400 device programs per step (122 top-level ttnn ops per layer) and
54 ms of device time of which ~27 ms is data movement (reshapes, layout and memory-config conversions, slices,
repeats, pads), against a 1.7 ms DRAM floor. Each op here removes one such chain; the measurement protocol is
`tests/unit/test_decode_roofline_lanes.py` (device ops/step, device ms/step) plus the lanes e2e for answers.

| op | replaces (per layer, decode) | status |
|---|---|---|
| `rope_decode` | q/k RoPE as `(getitem x2, neg, concat, multiply, add) x2, multiply x2` per rotation (~8 programs after the per-step cos/sin hoist) | done: 9 unit tests (pcc 0.99999x vs golden and composite, identity exact), 64.7 -> 7.2 us device per b32h16d256 call (13 -> 1 programs); opt-in `GEMMA4_ROPE_DECODE_CUSTOM_OP=1` in `attention/decode.py` |
| `geglu_routed_expert` | the shared `unified_routed_expert_ffn` edited for GeGLU (dispatch-prefill experts): now the only prefill expert path, TtRoutedExpert is the weight loader | done: bit-identical to the shared op with GeGLU, e2e dispatch PCC 0.99925 vs HF (128 / 1024 / 4224 tokens), same device time (541 us per 5633 rows per chip) |
| `rms_norm_decode` | the decode `to_memory_config -> rms_norm (width-sharded) -> sharded_to_interleaved` triple of every layer norm (7/layer) and the router's `rms_norm, mul, mul` | multi-core (11 cores on the 26B width) interleaved RMSNorm with a NoC partial-sum exchange, gamma and scalar folded; opt-in `GEMMA4_NORM_FUSED=1` (`tt/rms_norm.py`, `tt/router.py`) |
| `router_topk` | `softmax, topk (+fillpad), sum (fillpad+reduce), div, zeros_like, scatter (5), mul, matmul (expert select), max, to_layout, permute` (15/layer) | 32 RISC-V cores rank their user's row, core 0 gathers the local weight table `[E_loc, 32]` + the sparsity row; opt-in `GEMMA4_MOE_FUSED=1` (`tt/moe.py`, `tt/router.py`) |
| `geglu_decode` | experts `transpose, slice, slice, gelu, mul` (5/layer) and shared-MLP `slice, slice, gelu, mul` (4/layer) | `up * gelu(gate)` over the fused `[up | gate]` activation, exact GELU on fp32 dest; `GEMMA4_MOE_FUSED=1` (experts) / `GEMMA4_MLP_GEGLU_FUSED=1` (`tt/shared_mlp.py`) |
| `moe_combine` | `permute, mul, fast_reduce_nc` (3/layer) | per column tile the FPU column-broadcast products of the local experts accumulated on fp32 dest; reads only the selected experts' tiles; `GEMMA4_MOE_FUSED=1` (`tt/experts/decode.py::decode_forward_fused`) |
| `qkv_norm_rope` | `to_memory_config x3 (sharded -> DRAM), rms_norm x3, q/k RoPE (7 each or rope_decode), to_memory_config x2` (21/layer) | per-head norms + RoPE in place on the height-sharded head tensors (tensor-backed CBs), cos/sin rows taken straight from the per-step gather; opt-in `GEMMA4_QKV_NORM_ROPE_FUSED=1` (`tt/attention/decode.py`); `GEMMA4_QKV_NORM_ROPE_KERNEL` = 1 (SFPU-broadcast RoPE, page-order pack) or 2 (default: rope_decode's FPU row-broadcast RoPE on a bf16 normalized-column CB) |
| `moe_frame_prep` | on the decode MoE kernel rows (Galaxy lanes), the kernel's three dispatch inputs from the lane-gathered frame: `pad (hidden), pad (values), to_layout x3, concat (pad ids), mesh_partition x3, reshape x3, typecast, to_memory_config x3` (~15/layer) | one data-movement program per chip (32 cores, one token each): untilizes this row-chip's 32 frame rows into the L1 dispatch input, splits the packed routing into uint16 ids + height-sharded scores, synthesizes the frame's zero-weight padding rows; `GEMMA4_MOE_FUSED=1` with `GEMMA4_MOE_DECODE_KERNEL=1` (`tt/experts/moe_decode.py`); unit test on one chip, (2,4) / (8,4) through `test_moe.py -k "lanes and fused_router"` |
| `router_topk` (PACKED mode) | the kernel-row router tail `softmax, topk (+fillpad), sum (fillpad + reduce), mul, gather (+fillpad), div` (10/layer) and the lane gather's `typecast` + two of its three `all_gather`s | `op_packed`: every core writes its user's `[weights | global ids]` row of one `[1, 1, 32, 2k]` bf16 ROW_MAJOR tensor (no gather / semaphore); `GEMMA4_MOE_FUSED=1` on kernel rows (`tt/router.py::fused_sparse_routing`, `tt/moe.py`) |
| `concat_heads_decode` | `to_memory_config (-> height sharded), nlp_concat_heads_decode, sharded_to_interleaved (, slice)` (3-4/layer) | one data-movement program from the interleaved SDPA output to the interleaved o_proj input; opt-in `GEMMA4_CONCAT_HEADS_FUSED=1` (`tt/attention/operations.py`); work units per (column, head) on 32-64 cores (`GEMMA4_CONCAT_HEADS_PER_UNIT`), 27.8 us/layer with the one-unit-per-column split |
| `qkv_norm_rope_prefill` | prefill `rms_norm x3` with their flatten / unflatten `reshape` pairs (not views: the heads dim is tile-padded) and the two `rotary_embedding` programs (11/layer) | one program on the `[B, H, S, D]` head tensors: work unit = one `[32, D]` tile row of one head (32 positions) dealt round-robin over the grid, streamed through L1 (reader / compute / writer); the decode op's norm stages per unit, RoPE elementwise with the cos / sin tile rows of the same positions; fp32 dest + HiFi4 like the composite prefill norms; opt-in `GEMMA4_QKV_NORM_ROPE_PREFILL=1` (`tt/attention/prefill.py`, both call sites; KV-shared layers norm + rotate q only) |
| `fused_norm_decode` on prefill rows | the layer tail's 7 norms + 3 adds + 1 mul on `[1, 1, S, W]` prefill rows (11/layer) | the same kernels on one `N`-core exchange group per row tile (`4 N` cores at 128 tokens; the reader takes the gamma column base as its own arg); `tt/layer.py::_forward_tail_fused_prefill` = 4 programs around the shared MLP and the MoE block's prefill path; opt-in `GEMMA4_LAYER_NORM_FUSED_PREFILL=1` (with `GEMMA4_NORM_FUSED=2`) |
| `fused_norm_decode` | the decode-row norms and the adds / scalar around them: 8 norms + 3 adds + 1 mul per layer -> 5 programs (`post_attention + residual`, `[pre_feedforward, pre_feedforward_2, router norm*scale*H**-0.5]` from one exchange, `post_feedforward_1(mlp) + post_feedforward_2(experts)`, `post_feedforward + residual, * layer_scalar`, `input_layernorm`) | 22 cores (`GEMMA4_FUSED_NORM_CORES`), each 4 column tiles: FPU x^2 + row reduce, a 128-byte partial-sum column exchange, FPU column / row broadcasts (gamma as a row-replicated tile, dest reused for gamma and the residual), one bf16 pack; `GEMMA4_NORM_FUSED=2` (`tt/rms_norm.py`, single norms) + `GEMMA4_LAYER_NORM_FUSED=1` (`tt/layer.py`, the five-program layer tail; needs `GEMMA4_MOE_FUSED=1`) |

**Defaults on the 4-chip Blackhole 26B build (2026-10-23).** On a Blackhole MoE build on the (1,4) or (4,1) mesh the
fused decode step is the default (`tt/mesh_orientation.py::decode_fusion_defaults`, recorded by the model build, read
through `decode_flag`): `rms_norm_decode`'s successor `fused_norm_decode` (`GEMMA4_NORM_FUSED=2`, `GEMMA4_LAYER_NORM_FUSED=1`),
`router_topk` + `geglu_decode` + `moe_combine` (`GEMMA4_MOE_FUSED=1`), the shared-MLP `geglu_decode` (`GEMMA4_MLP_GEGLU_FUSED=1`),
`concat_heads_decode` (`GEMMA4_CONCAT_HEADS_FUSED=1`), `qkv_norm_rope` in its in-place form (`GEMMA4_QKV_NORM_ROPE_FUSED=1`),
the call-site switches `GEMMA4_QKV_L1_OUT=1`, `GEMMA4_SOFTCAP_FUSED=1`, `GEMMA4_HEAD_NORM_4D=1` and the explicit 1D decode
matmul configs (`GEMMA4_MLP_DECODE_1D`, `GEMMA4_ATTN_DECODE_1D`, `GEMMA4_ROUTER_DECODE_1D`). Setting any of these env vars
still overrides the default (`0` restores the old path). Gate: 64-position HF logits b1 52/64 top-1 (decode PCC 0.98481),
b32 50/64 (0.98558); traced step at 32 distinct users 35.7 -> 30.2 ms (28.0 -> 33.1 tok/s/user), b1 23.8 -> 23.0 ms
(device sampling, line 31..28). Other meshes (the Galaxy lane engine), Wormhole and the dense 31B / 12B keep the opt-in
defaults; `rope_decode` (`GEMMA4_ROPE_DECODE_CUSTOM_OP`) and the kernel-row ops stay opt-in everywhere.

Call-site switches without a kernel: `GEMMA4_QKV_L1_OUT=1` (the wqkv projection writes L1, dropping the DRAM -> L1 copy of
the Blackhole head-split workaround), `GEMMA4_SOFTCAP_FUSED=1` (`ttnn.softcap` for the `mul, tanh, mul` of the lm head),
`GEMMA4_MOE_DECODE_TILE_TAIL=1` (the kernel-row frame tail gathers in TILE layout: no `untilize` before / `tilize` after the
two output all-gathers, `tt/experts/moe_decode.py`). `GEMMA4_MLP_GEGLU_FUSED` also serves the lane engine's gathered shared-MLP
rows (4 lanes x 32, up to four row tiles).
Measurements (traced ms/step, program counts) are in `_logs/a9_summary.tsv` / the 2026-10-23 report.

Layout of an op folder:

```
custom_ops/<op>/op.py        # class <Op>: golden(torch) + op(ttnn tensors) -> ttnn.generic_op(...)
custom_ops/<op>/kernels/     # reader_<op>.cpp, compute_<op>.cpp, writer_<op>.cpp
tests/unit/custom_ops/test_<op>.py
```

Kernel traps found while writing these (2026-10-23, Blackhole, `ttnn.generic_op`):

* An FPU write (`mul_tiles*`, `copy_tile`) to a dest tile that an earlier SFPU op of the same acquire window reads
  corrupts the result (`moe_combine`: wrong sums from 3 experts on). Issue every unpack copy / FPU op of a window
  before its first SFPU op, or pass the running value through an L1 CB between windows.
* `pack_tile(idst, cb, index)` writes sequentially from the reserved block regardless of `index`; an out-of-order
  index needs `pack_tile<true>(...)` (`qkv_norm_rope`: pages 0,4,1,5 landed in 0,1,2,3). Pack in page order.
* The math core (TRISC1) has no `cb_interface`: a compute kernel cannot read a data-dependent count from a CB
  (`moe_combine`: fixed expert loop, the reader pushes zero weight tiles for the experts nobody selected).
* `get_read_ptr` is a dataflow-only free function; compute kernels use `CircularBuffer::get_read_ptr()`.
* A kernel-side `union F32` collides with `dprint.h`; and the ROW_MAJOR / TILE row tile of the position-gathered
  cos / sin can be copied per user by the reader (32-byte face rows), so no per-step transpose / slice / repeat.
