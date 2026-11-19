# Untilize once before the exact convolution pipeline

The current custom convolution repeatedly gathers compact TILE input into a
contiguous per-user channel block. For B32 it issues 320 tasks × 64 small token
reads and performs 320 × 512 RISC-V word copies. The published profile88
convolution takes about 87 us; that profile still uses the earlier hidden norm,
so its total must not be confused with the parent's later exact82 timing.

This candidate replaces the repeated token gathers with one native multicore
TILE-to-ROW_MAJOR conversion, then one contiguous 2048-byte token read/task.
Convolution arithmetic and the verified two-slot pipeline protocol are unchanged.
The additional conversion must be measured; no performance improvement is claimed.

## Immutable artifacts

- `candidates/conv_decode_sfpu_untilize_pipeline2.py`: standalone helper.
- `candidates/make_conv_untilize_candidate.py`: integration generator accepting
  either the original custom-conv call or the pipeline2 call.
- `candidates/conv_untilize_on_exact82.py`: generated integration on exact82.
- `candidates/check_conv_untilize_cpu.py` and `conv_untilize_cpu.json`: source
  and address equivalence proof.

Existing helpers, main decoder and shared source were not modified. The new
source globals are `_CONV_RM_PIPE2_*`; the integrated Python function is
`conv_decode_sfpu_untilize_pipeline2`. The generator refuses overwrites and
requires exactly one known call site.

The standalone public API remains
`conv_decode_sfpu(token, history, taps, output=None, cores=None)` and accepts the
same compact BF16 TILE token `[1,B,10240]`, B=1..32. Inside the helper,
`ttnn.to_layout(token, ROW_MAJOR_LAYOUT)` produces a temporary raw token tensor.
It is registered with the generic op alongside history/taps/output. History
continues to use its stable in-place allocation, and caller-provided output
remains supported. There is no host tensor conversion or read.

## Source rationale and exact mapping

`ttnn/cpp/ttnn/operations/core/to_layout/to_layout_op.cpp:126` selects multicore
untilize. When padding changes, such as B<32, the same implementation calls
`untilize_with_unpadding` with the logical tensor end. This is also the layout
conversion used by the original primitive `_batched_conv` path.

The resulting row-major page is one logical token row: 10240 BF16 values, or
20480 bytes. The new reader uses `page_id=user`, `offset_bytes=chunk*2048`, and
reads 2048 bytes directly into CB3. Both addresses are 64-byte aligned. The old
reader's final packed physical sequence was exactly channels
`chunk*1024 .. (chunk+1)*1024-1` for that user; it was not a spatially tiled
matrix permutation at this boundary. The row-major contiguous read produces
that exact sequence. All operands therefore retain the same common physical
permutation inside the compute tile.

The CPU proof reconstructs every old source read, pair-row alignment adjustment,
and packed-word selection using distinct logical element labels. All 32 users
and all ten chunks match the new contiguous row-major address sequence exactly.
The new compute and writer kernel strings are literally identical to pipeline2.
The reader-private scratch CB8 and its RISC-V copy loop disappear. Total CB
allocation decreases from pipeline2's 44 KiB/core to 40 KiB/core.

At B32, the custom reader removes 20,480 small token reads and 163,840 RISC-V
word-copy iterations, replacing them with 320 contiguous token reads. Counting
native untilize's full input read/output write, aggregate token traffic grows
from approximately 1.25 MiB to 1.875 MiB. The hypothesis trades roughly 640 KiB
of additional bulk traffic and one dispatch for substantially less per-read
and RISC-V overhead. Untilize may itself be expensive or allocate awkwardly;
only measurement can decide the tradeoff.

## Host validation completed

Python syntax compilation passes for helper and generated decoder. The CPU
address/source checker passes, and the integration generator successfully
created the frozen exact82-based candidate.

```sh
python3 models/autoports/qwen_qwen3_6_27b/doc/optimized_decoder/candidates/check_conv_untilize_cpu.py > models/autoports/qwen_qwen3_6_27b/doc/optimized_decoder/conv_untilize_cpu.json
python3 models/autoports/qwen_qwen3_6_27b/doc/optimized_decoder/candidates/make_conv_untilize_candidate.py --base exact82.py --output conv_untilize_on_exact82.py
```

No hardware was opened. The prior pipeline2 scheduling proof applies unchanged
because compute/writer protocol is identical and token publication still follows
the reader's complete NoC barrier.

## Parent-owned device validation

```sh
python -m models.autoports.qwen_qwen3_6_27b.tests.run_conv_precision_probe --candidate models/autoports/qwen_qwen3_6_27b/doc/optimized_decoder/candidates/conv_decode_sfpu_untilize_pipeline2.py --require-baseline-exact --output models/autoports/qwen_qwen3_6_27b/doc/optimized_decoder/conv_untilize_probe.json
```

Also exercise `--cores 1 --batch 3 --steps 3` for repeated ring wraps, B1/B3
untilize unpadding, changed input/output reuse, watcher and trace execution.
The unchanged 512-step accuracy gate and warmed decoder latency remain required.
No old successful result has been attributed to this new candidate.

## Other topology findings

The residual add preceding post-attention hidden norm is about 4 us in the
available profile. Fusing it into exact-tree norm pass1 would require preserving
the BF16 residual rounding boundary and publishing the residual tensor for the
final MLP add. Its limited saving does not justify prioritizing it over the
convolution gather bottleneck.

A larger independent opportunity is the post-GDN epilogue: native BF16 RMSNorm,
BF16 gamma multiplication, a material reshape from `[1,B*48,128]` to
`[1,B,6144]` (about 30 us), z slicing, FP32 casts, and FP32 SiLU/gating followed
by BF16 output. Approximately 56 us after RMSNorm is potentially fusible.
The recurrence investigator confirmed that area remains free; the correction
pair investigator does not change norm/residual interfaces.

That epilogue is not implemented here. Source inspection shows the native BF16
gamma multiply uses BF16 destination mode, while FP32 gating uses FP32 mode;
replacing both with an arbitrary FP32 SFPU expression would not establish exact
intermediate rounding. A controlled next candidate can retain native RMSNorm
and gamma multiplication, fuse only layout conversion plus FP32 gating, and
require bit-exact comparison before folding the BF16 gamma boundary as well.
This is a separate hypothesis, not a claimed property of the untilize candidate.
