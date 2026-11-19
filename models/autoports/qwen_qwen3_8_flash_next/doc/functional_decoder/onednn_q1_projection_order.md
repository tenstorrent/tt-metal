# Q1 indexer projection order

The recorded CPU backend uses five independent K=512 partials for logical Q1,
whereas Q2 through Q32 use K blocks 1024, 1024 and 512. Each partial retains the
same high-member-then-low-member FP32 pair accumulation described in
[the original source investigation](onednn_projection_order.md), followed by
ordered FP32 additions of partials and final BF16 rounding. The distinction is
based on input geometry, never token identity, output column or value.

The [608-case shape check](qsa_blocked_projection_shapes_cpu_v1.json) covers
19 real input starts and every Q1 through Q32. K=1024 matches every BF16 output
for Q2 through Q32. Q1 has three differences among 12160 outputs; notably token
58531 column244 differs from its original Q32 projection. A separate
[fixed-block Q1 sweep](qsa_q1_blocks_cpu_v1.json) compares ten block sizes: only
K=512 matches all 12160 outputs. Both reports preserve every failure. Their
literal calls repeat bitexactly, and the raw verbose log identifies the same
`brg_matmul:avx512_core_bf16` implementation across all shapes.

The pinned oneDNN source explains the distinction:

1. `matmul_helper_t::transA()` returns nontransposed only when the K stride is one
   **and M is greater than one**. M=1 therefore returns transposed.
   [Helper, lines50–53](https://github.com/uxlfoundation/oneDNN/blob/f1d471933dc852f956fd05389f9313c7148783d5/src/cpu/matmul/matmul_utils.hpp#L50).
2. For a singleton M dimension, plain and transposed descriptors can both
   match. The configuration explicitly resolves this ambiguity with
   `helper.transA()` and retains the resulting `src_tag`.
   [Tag resolution, lines568–583](https://github.com/uxlfoundation/oneDNN/blob/f1d471933dc852f956fd05389f9313c7148783d5/src/cpu/x64/matmul/brgemm_matmul_utils.cpp#L568).
3. The AVX512 K-block heuristic uses `src_tag`: K>1024 extends the default block
   from512 to1024 only for a nontransposed tag.
   [Heuristic, lines866–870](https://github.com/uxlfoundation/oneDNN/blob/f1d471933dc852f956fd05389f9313c7148783d5/src/cpu/x64/matmul/brgemm_matmul_utils.cpp#L866).
4. Separately, `treat_A_as_plain` permits nontransposed memory access for
   canonical M=1 inputs. It sets effective `transposed_A`; it does not replace
   the tag consumed by the K-block heuristic.
   [Memory-access interpretation, lines1571–1578](https://github.com/uxlfoundation/oneDNN/blob/f1d471933dc852f956fd05389f9313c7148783d5/src/cpu/x64/matmul/brgemm_matmul_utils.cpp#L1571).

[Source and evidence hashes](onednn_sources/q1_provenance.json) bind the pinned
upstream files, retained line-numbered excerpts, and actual CPU results. This
is a source-supported numerical control for the recorded backend and shapes,
not a claim of device implementation or full decoder acceptance. The separate [full262144-row Q32 projection check](qsa_blocked_projection_full_cpu_v1.json)
completed with all167772160 BF16 projection outputs bitexact to literal HF.
Both full projection arrays and per-row mismatch counts are retained externally
with hashes in that report. Device implementation and full decoder gates remain
separate requirements.
