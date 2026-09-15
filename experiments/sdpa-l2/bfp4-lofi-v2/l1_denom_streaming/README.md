# Local BF16 denominator, FP32 streaming state

Isolated research prototype; no shared/frozen compute header was modified.
The private header is a copy of `../streaming/compute_streaming.hpp` with
only a configuration guard and the optional local-denominator branches added.
There are no on-device results yet.

## Fixed controls

Both drivers use Q256/K512/D128, FP32 DST and recurrent numerator/denominator,
FP32 score/P alias CB6/7, BF16 maxima and final output, native one-pass exp,
LoFi QK/PV, device Q-RNE7/BF16 and K/V-RNE5/native-BFP8 preprocessing.
There is no additional P rounding. Q has two slots, K and V each have the
same one slot as the existing FP32 control. Readers/writers include the
unchanged fullchip-chain or resident-input kernels.

`--denominator matched` retains the original 128-P-tile LoFi P-times-ones
denominator, consuming the same effective P significand as PV.

`--denominator l1_bf16` additionally packs each live FP32 exp DST tile to
BF16 scratch CB20, accumulating the 16 K tiles lane-wise. Each Q tile row
has its own scratch tile. The first tile overwrites; subsequent tiles use
L1 accumulation. After the complete-P drain barrier, eight scratch tiles
are published and consumed by two four-row HiFi4 matmuls against the existing
column-identity tile. These eight column-reduced FP32 results replace only
the current chunk's sum. Existing FP32 online recurrence and normalization
remain unchanged. Scratch is popped/wrapped and reset every K512 chunk.

HiFi4 is deliberate: ordinary HiFi2 phase0/1 does not retain the low bits
of logical-left/SrcB BF16 when logical-right/SrcA is exactly zero/one.
A specialized phase0/2 reduction could be tested later; it is not used here.

Both controls allocate the same extra eight BF16 scratch tiles: 16,384 B.
Total CB allocation is 1,228,800 B/core, leaving 344,064 B of raw 1.5-MiB L1
before program/firmware/allocator overhead. No Q/K/V or score capacity changes.

## Numerical and performance tradeoffs

The local sum is intentionally **unmatched** to PV: BF16 local packing and
addition see different bits from LoFi's seven-significant-bit P consumption.
Native exp and all matmuls are otherwise identical. This can introduce a
systematic denominator/gain error; no precision equivalence is claimed.
Local rounding spans 16 positive additions per lane, not all context chunks.

The scheme replaces 128 LoFi P-tile reductions with eight HiFi4 reductions,
but adds 128 BF16 packs and repeated pack-format changes on the exp path.
Instruction savings therefore do not establish a latency improvement.
L1 accumulation is disabled and FP32 width4 packing restored before returning
to QK. After the scratch reduction, SrcB is restored to FP32 P explicitly:
the later PV fast reconfiguration assumes that matched-denominator state.

## Qualification commands (run by the device owner)

Small correctness, then change `matched` to `l1_bf16` with a fresh label:

```sh
python experiments/sdpa-l2/bfp4-lofi-v2/l1_denom_streaming.py --label l1denom-smoke-matched-v1 --denominator matched --length 1024 --heads 1 --cores 1 --sample-rows 1024 --check-preprocess --iters 0
python experiments/sdpa-l2/bfp4-lofi-v2/l1_denom_resident.py --label l1denom-resident-smoke-matched-v1 --denominator matched --q-repeats 1 --k-chunks 2 --iters 0
```

After both modes pass, repeat resident with default q-repeats16/k-chunks512,
then the fullchip driver at N32768/H10/110 cores. Repeat normal and constant_v
at minimum to separate general quantization error from coherent gain drift.
Every run has a finite-output check and an L2 gate before timing (`--max-l2`,
default10 percent for exploratory qualification). Original BF16 Q/K/V are
the FP64 reference. Fullchip timing separately reports preprocessing and
combined cost; resident timing excludes preprocessing and has no recurring
input movement, only per-invocation input initialization and final output.

The resident reference uses all256 Q rows against the resident512 tokens;
repeating identical K/V chunks leaves exact normalized attention unchanged.
It is not a distinct-256K-key accuracy test. Fullchip uses actual distinct
keys and explicitly reported sampled query rows. Both pin their source files
before execution and check them again afterward.

Static validation: Python AST/whitespace and source-path checks pass; host
preprocessing of both header branches passes with includes stripped. This is
not a Tenstorrent JIT build or hardware validation. No device job was launched
by the implementation agent.
