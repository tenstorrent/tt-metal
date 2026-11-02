# Ternary prefill (BONSAI_NATIVE=pq2_0)

The A3 decode op is M=1. Prefill (M = 128..2048 rows per chunk) reads the same resident A3 weights
(bfp2 codes, A2b 256 B tiles or 320 B Bfp2_b; fp32 `scale_rows` scales). No dense copy of a ternary site
is loaded and no extra cache is written. Every `TernaryLinear` site and the fused MLP go through
`prefill.py`:

    y = H(D x) @ (codes * scales)

| Step | S <= 128 | S > 128 |
|---|---|---|
| rotate x' = H(D x) | stock: transpose, batched matmul with `[nb,1024,1024]` blocks | `rot.py` kernel when S*K >= 10M, else stock |
| project | `pmm.py` (A3 at M>1): dequant in L1, no scratch, x' multicast from one sender core | `dequant()` into a per-call bfp8 DRAM `[K,N]`, then `ttnn.linear` |
| matmul config | (in pmm) | N <= 5120: dense gdn_out config (`create_prefill_mlp_matmul_program_config`, 11 cols); wide N: `wide_pc` (8x10, out_block_w = per_core_N) |

Hooks: `qwen36/tt/mlp.py` (rows > 32), `gdn/tp.py` (`_project_qkvzab`, `_row_proj`),
`ttnn_gated_attention.py` (fused qkv, o_proj) and `attention/prefill.py`, which passes the ternary fused qkv.
The LM head prefill uses the dense head; only the last row is read.
`drop_dense()` (prefill.py, called from layer.py) frees the dense weights each ternary site replaced: 176 tensors, 8.32 GiB of DRAM.

## Numbers (chip 3, text_demo, 5-run medians, three arms interleaved, commit 28e517fe401, A's defaults at 97a1ccf4495)

| Stack | traced_128 TTFT | traced_4k TTFT | Decode (traced_4k) |
|---|---|---|---|
| Dense six-flag control | 0.168 s | 4.326 s | 79.1 ms/token (12.6 tok/s) |
| Dense full (B's flags + PF_GATEUP) | 0.200 s | 3.201 s | 73.0 ms/token (13.7 tok/s) |
| Ternary PQ2_0, all sites | 0.164 s | 3.135 s | 34.4 ms/token (29.1 tok/s) |

Agreement:
- text_demo accuracy_512 (512-token prefill through this path, 512 teacher-forced tokens): ternary 96.29/100.00 at these defaults, dense control 93.95/100.00.
- test_decode_e2e against the llama.cpp PQ2_0 reference: 95.51/100.00 on the scored half.

Op level, S=128, ms: pmm vs dequant+matmul:

| Site | pmm | dequant+matmul |
|---|---|---|
| gate+up | 0.61 | 1.74 |
| down | 0.54 | 1.24 |
| gdn_in | 0.51 | 1.02 |
| gdn_out | 0.39 | 0.72 |

PCC 0.99999 vs float64, against 0.99986 on the bfp8 scratch path.

At S=2048 the rotation kernel against the stock rotator, ms: K=17408 0.87 vs 3.26; 5120 0.75 vs 0.99.

## TTFT by stage (traced_4k, 2642 tokens; chip 3, 2026-11-02, bd0694ce6dd = A's pq2_0 0aebbadcf94)

Wall times, from test_prefill_profile.py with one traced prefill per process and no profiler:

| Prompt | What runs | Wall |
|---|---|---|
| 640 | one traced 2048-row chunk replay (masked) + logits | 1.129 s (chip 1, A 05:37) |
| 2642 | the same replay + the eager 594-token tail | 2.809 s (chip 3, 09:44; 2.802 s on chip 1; cs5 TTFT 2.805 s) |

So the eager tail costs **~1.68 s of the 2.81 s TTFT**, against ~1.13 s for the 2048-row chunk.

Device kernel time by stage comes from one tracy run (`agentC/profiles/pf2642_1102-0953c`, `agentC/pf_segments.py`). The device profiler buffer filled during the warmup prefill, so the CSV covers that pass only. It is eager and layer-outer: each GDN layer runs a 2048-row chunk and then a 608-row chunk, and the 12 attention layers run 2656 rows at once. Those are the same kernels and program configs as the traced path, except attention, which the traced path runs paged, per chunk.

| Device ms | 2048-row chunk (36 GDN layers + MLPs) | 608-row tail (36 GDN layers + MLPs) | Attention layers + MLPs, 2656 rows (12) |
|---|---|---|---|
| Projection matmuls (bfp8 dequantized weights) | 544.7 | 158.6 | 227.6 |
| Ternary dequant to the bfp8 scratch | 77.0 | 76.6 | 24.0 |
| Rotation (KRotator kernel / stock 1024x1024 bmm) + all transposes | 45.3 + 25.8 | 7.7 + 28.9 + 14.6 | 18.6 + 14.4 |
| GDN core (ChunkGdn + conv1d) / SDPA | 38.5 | 15.0 | 29.1 |
| Everything else (eltwise, norms, reshapes, slices, tilize) | 177.7 | 39.8 | 121.2 (71.8 of it ReshapeView in the non-paged path) |
| **Total** | **909.0** | **341.2** | **434.9** |

Recommendation, largest lever first:
1. **Trace the tail.** The 608-row tail runs ~0.34 s of GDN-layer device work, plus its share of attention (~0.1 s), but costs ~1.68 s of wall. That leaves **~1.2 s of host dispatch for ~2,800 eager ops**, about 0.4 ms per op. A second captured trace for the tail bucket (e.g. 640 rows, or a few tail buckets) should take traced_4k TTFT from ~2.8 s to ~1.6 s. This is model-level, not ternary-specific, so the dense stacks gain the same.
2. **Ternary dequant costs the same at 608 rows as at 2048 (~77 ms per pass).** It's bound by weight size, not S. For the tail that's 22% of its device time. A matmul that dequantizes in1 into L1 (pmm, generalized past S=128 with 2D multicast) removes it, plus the tail's stock-rotation bmm (~29 ms): up to ~0.1 s.
3. The projection matmuls (~0.93 s of device time per prefill) run near the stock-op limits for bfp8 weights. They aren't a lever without a new matmul.

## Checks and tools

- `check_prefill.py` (MLP), `check_linear.py` (every site), `pmm.py` and `rot.py` (self-checks): PCC against float64 and against the A3 decode op on row 0, plus wall ms.
- `pftime.py`: `BONSAI_PF_TIME=1` gives per-site in-model prefill times (first two eager passes), dense or native.
- Knobs:
  - `BONSAI_PF_PMM=0`: turn pmm off.
  - `BONSAI_PF_KROT=0`: stock rotation only.
  - `BONSAI_PF_W=bfp4`, `BONSAI_PF_GU=bfp4`: bfp4 scratch. Null result; it costs PCC.

## Traps

- **Matmul compute kernels need `compute_kernel_hw_startup<SrcOrder::Reverse>`**, because in1 goes to SrcA. Without it the rotation kernel's output was a constant per row.
- **Switching ops inside one compute kernel needs an explicit `reconfig_data_format`** (and `pack_reconfig_data_format`) at each switch. Without it, pmm read its bf16 W tiles with the dequant step's formats and produced values around 1e38.
- **A2b code tiles have no exponent section.** Force exponent 127 on unpacker 0 for the code tiles, as A's kernel does. In pmm, switch it off before the bf16 matmul and the fp32 partial reload, which also go through SrcA.
- **`cpu_ref.pcc` returns 1.0 when the data contains inf/NaN** (NaN denominator). Check `isfinite` first.
- **PCC can't see a global scale error.** Under A's fp16 scales (SCALE16), reprogramming SrcB to Float16 in compute decoded the fp16 bits with an 8-bit-exponent bias. Every prefill output came out ~1e-33 too small with the right shape, so every PCC-only check passed. `cpu_ref.pcc` even returns 1.0 for an all-zero output. A's in-model magnitude check found it (01:34). Fix: the readers convert the needed fp16 scale rows to fp32 (`kernels/scale_cvt.h`), and every check now also reports max\|y-ref\|/max\|ref\|.
- **Op checks on freshly built objects don't cover the model path.** `tp1/test_prefill_inmodel.py` runs the model's own TernaryLinear/TernaryMLP objects; run it, plus a traced_4k demo text check, for any prefill-affecting change.
- **`ttnn.matmul` output dtype defaults to in0's dtype.** The rotation with bfp8 H blocks came out bfp8 until `dtype=bfloat16` was set.
- **ttnn's auto matmul config degenerates at 2048 rows for wide N:** gate/up 12.1 ms against 3.8 with `wide_pc`. The tuned qwen36 `prefill_progcfg` doesn't fit gate/up in L1.
- **The dense gdn_out config, used for wide N, overflows L1 at S=2048 and clashes with resident L1 buffers at S=640.** So it's used for N <= 5120 only.
- **On Blackhole, BRISC (writer-config) kernels run on NOC1.** A NOC1 multicast goes from the high corner to the low one; the reverse order never completes its acks and hangs the chip. The non-Tensix NOC columns 8-9 inside a rectangle are fine.
- **Under BONSAI_NATIVE, decode-only L1 placements must stay decode-only.** A prefill-sized activation in L1 collides with the prefill programs' static CBs.
- **traced_4k prefill is one traced 2048 chunk plus an eager tail** (the 594-token remainder, padded to 640 rows). traced_128 is an eager masked bucket.
