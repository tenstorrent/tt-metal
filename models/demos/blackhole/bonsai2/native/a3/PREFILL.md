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
- **`ttnn.matmul` output dtype defaults to in0's dtype.** The rotation with bfp8 H blocks came out bfp8 until `dtype=bfloat16` was set.
- **ttnn's auto matmul config degenerates at 2048 rows for wide N:** gate/up 12.1 ms against 3.8 with `wide_pc`. The tuned qwen36 `prefill_progcfg` doesn't fit gate/up in L1.
- **The dense gdn_out config, used for wide N, overflows L1 at S=2048 and clashes with resident L1 buffers at S=640.** So it's used for N <= 5120 only.
- **On Blackhole, BRISC (writer-config) kernels run on NOC1.** A NOC1 multicast goes from the high corner to the low one; the reverse order never completes its acks and hangs the chip. The non-Tensix NOC columns 8-9 inside a rectangle are fine.
- **Under BONSAI_NATIVE, decode-only L1 placements must stay decode-only.** A prefill-sized activation in L1 collides with the prefill programs' static CBs.
- **traced_4k prefill is one traced 2048 chunk plus an eager tail** (the 594-token remainder, padded to 640 rows). traced_128 is an eager masked bucket.
