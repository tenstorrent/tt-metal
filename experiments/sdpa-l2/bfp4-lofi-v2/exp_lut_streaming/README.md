# Native-grid LUT exp refinement

New drivers and wrappers only. The shared `../streaming/compute_streaming.hpp`,
frozen common helpers, native exp, input preprocessing and dataflow sources
are not changed. The experiment's `../exp_lut.hpp` is owned separately by the
main agent and must be present before either driver runs.

Both modes use Q256/K512/D128, LoFi QK/PV, FP32 DST, FP32 recurrent state,
FP32 score/P CB6/7 alias, BF16 maxima and final output. Device input preparation
is Q-RNE7/BF16 and K/V-RNE5/native-BFP8. Q has two slots; K/V each have one,
unchanged from the FP32 control. No extra scratch CB, P rounding or new input
movement is introduced. Total CB allocation is the baseline1,212,416 B/core.

Default mode is native exp: `SDPA_LOFI_NATIVE_EXP` is always set, and the LUT
header falls through to the original native helper without extra calls.
`--lut-exp` additionally defines `SDPA_LOFI_LUT_EXP`. The header preincludes
native exp under a temporary helper-name rename and defines the original-name
wrapper as native8-bit grid followed by a two-segment FP16 LUT refinement.
Compute wrappers include frozen common first, LUT header second, shared streaming
third. This intercepts existing calls without changing the streaming header.
The denominator remains LoFi P-times-ones, consuming the same stored/effective
P as PV in both modes. LUT accuracy/performance is experimental, not assumed.

## Device-owner commands

All1024 output rows against original BF16 FP64 reference:

```sh
python experiments/sdpa-l2/bfp4-lofi-v2/exp_lut_streaming.py --label explut-native-smoke-v1 --length 1024 --heads 1 --cores 1 --sample-rows 1024 --check-preprocess --iters 0
python experiments/sdpa-l2/bfp4-lofi-v2/exp_lut_streaming.py --label explut-refined-smoke-v1 --lut-exp --length 1024 --heads 1 --cores 1 --sample-rows 1024 --check-preprocess --iters 0
```

Resident smoke and timing:

```sh
python experiments/sdpa-l2/bfp4-lofi-v2/exp_lut_resident.py --label explut-native-res-smoke-v1 --q-repeats 1 --k-chunks 2 --iters 0
python experiments/sdpa-l2/bfp4-lofi-v2/exp_lut_resident.py --label explut-refined-res-smoke-v1 --lut-exp --q-repeats 1 --k-chunks 2 --iters 0
python experiments/sdpa-l2/bfp4-lofi-v2/exp_lut_resident.py --label explut-native-res-perf-v1
python experiments/sdpa-l2/bfp4-lofi-v2/exp_lut_resident.py --label explut-refined-res-perf-v1 --lut-exp
```

Resident defaults are q-repeats16/k-chunks512. Exact repeated-KV equivalence
allows comparison of all256 Q rows against the resident512 original BF16
tokens. This is not an accuracy test with262K distinct keys. Preprocessing is
excluded from resident timing; initialization reads and final-output write are
included. Fullchip uses distinct keys and separately reports preprocessing,
attention, and combined timing. Both check finite output, L2 before timing
(`--max-l2`, default10 percent), replay equality, and source hashes before/after.

Static validation: Python AST, whitespace and baseline CB/config/reader-argument
comparisons passed. No JIT compilation or device jobs were run by this agent.
