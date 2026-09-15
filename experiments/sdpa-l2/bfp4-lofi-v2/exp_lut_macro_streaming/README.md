# Native-grid LUT LOADMACRO prototype

New isolated drivers/header only; existing `exp_lut.hpp`, streaming compute,
input quantizers and dataflow are unchanged. No device compilation, correctness
or timing has been run by the author. Parent owns hardware qualification.

Parent-run smoke evidence is now available and inspected locally:
[raw LUT](../lut-macro-raw-smoke-v1.json) and
[macro LUT](../lut-macro-macro-smoke-v1.json), N1024/H2/4cores, all1024 query
rows per head. Both output SHA256 values are
`612a059d779aed90461580eb7fd3a592bd96a32f130997387c6e6dab0fd10924`;
both have L2 2.2506252607%, PCC0.9997465773 and gain0.9995517228.
This validates output equality for that smoke, not arbitrary distributions.

Additional parent-run normal-input records, inspected locally:

| Scope | Raw LUT | Macro LUT | Unchanged L2 |
|---|---:|---:|---:|
| Resident Q256/K512,16 Q repeats/512 K chunks, TFLOP/core | 1.14860 | 1.20356 | 2.24428% |
| N32768/H10/110cores, attention-only TFLOP/s | 118.7377 | 123.9697 | 2.20760% |
| Same32768 case, combined preprocessing+attention TFLOP/s | 115.9285 | 121.0647 | 2.20760% |
| N262144/H10/110cores, attention-only TFLOP/s | 123.3705 | 128.8124 | 2.20918% |
| Same262144 case, combined preprocessing+attention TFLOP/s | 123.0107 | 128.6645 | 2.20918% |

Each raw/macro pair has identical complete-output SHA256, including the two
fullchip cases (accuracy reference uses the recorded sampled Q rows). Useful
FLOPs exclude preprocessing arithmetic even in combined timing. Resident data
repeats512 keys, not262144 distinct keys. These measurements show approximately
4.4–4.8% throughput improvement in these cases; they do not claim recovery of
all native-exp throughput or qualification across all distributions.
Evidence: [resident raw](../lut-macro-raw-resident-v1.json),
[resident macro](../lut-macro-macro-resident-v1.json),
[32768 raw](../lut-macro-raw-32768-v1.json),
[32768 macro](../lut-macro-macro-32768-v1.json),
[262144 raw](../lut-macro-raw-262144-v1.json),
[262144 macro](../lut-macro-macro-262144-v1.json).

Controls in both `exp_lut_macro_streaming.py` and `exp_lut_macro_resident.py`:

- No flags: unchanged native exp.
- `--lut-exp --raw-lut`: unchanged `exp_lut.hpp` ten-instruction replay.
- `--lut-exp`: new eight-instruction LOADMACRO replay.

Q256/K512/D128, double Q and single K/V slots, all CB capacities/formats,
LoFi QK/PV, FP32 score/P/DST/recurrence, BF16 maxima/output, and matched-P
denominator remain identical. Inputs remain Q RNE7/BF16 and K/V RNE5/native
BFP8 with the original non-precise pack setting. Fullchip times include input
preprocessing separately and combined; resident timing excludes preprocessing.

## Unchanged mathematics

Native approximate exp writes signed grid value `g`. The refiner computes
`m=SETEXP(g,127)`, `f=SFPLUTFP32_TABLE1(abs(m))`, then `g*f` exactly as the raw
refiner, with slopes `0x2f59aee8` and intercepts `0x3a1c3c5f`. No new rounding,
range clamp, gain adjustment, polynomial or probability preprocessing occurs.
The multiplication operands commute for the finite grid values in scope;
caller pack ReLU still handles negative underflow grid results.

## Macro schedule and ownership

Sequence1 (`0x00000d04`) loads L3, schedules SETEXP127 at delay0 and TABLE1 LUT
at delay1, writing L3. Sequence2 (`0x13008600`) reloads the original grid into
L0, schedules `L3*L0` at delay0 and stores L0 at delay2 to the captured address.
L1/L5 are coefficient registers. Native programmable constants L12–14 are
untouched: VD12–14 instructions below program the instruction-template backdoor.

| Issue slot | Issued instruction | Scheduled useful operation |
|---|---|---|
| 0 | Load macro1, vector A to L3 | — |
| 1 | NOP | SETEXP A |
| 2 | NOP | LUT A |
| 3 | Load macro2, A to L0 | — |
| 4 | Load macro1, B to L3 | MUL A (reads old L3 before load completes) |
| 5 | NOP | SETEXP B |
| 6 | NOP | LUT B and STORE A |
| 7 | Load macro2, B to L0, advance DST | — |
| 8 | Next pair's macro1 | MUL B |
| 10 | Next pair's LUT | STORE B |

MAD producers and consumers have an intervening cycle. No issued arithmetic
instruction competes with a delayed macro operation. Three final SFPU NOPs
drain the last store. Replay slots8–15 replace only the refiner; native replay
slots0–7, Sequence0 and template3 remain unchanged. Native MAD/round templates1/2
are restored after every refinement; template0 and sequences1/2 are unused by
the required unclamped native grid. Native init reinitializes grid state at each
QK/exp phase. Macro setup repeats per call for safety; its extra setup cost is
not included in the 8-versus-10 replay count. For128 vectors, the loop saves128
issued instructions before setup overhead. This is not a measured speedup.

The two key hardware points are SFPLUTFP32's macro destination override to L3
and the same-cycle old-L3 read/new-L3 load boundary. The parent-run smoke above
exercises both successfully. Continue requiring raw versus macro **identical
output SHA256**, not merely a similar L2 or gain, for new qualification cases.

## Parent-run smoke and timing

Run from repository root with the existing configured TTNN environment/cache.
Use a fresh label each time. First run both full-output smokes:

```bash
python experiments/sdpa-l2/bfp4-lofi-v2/exp_lut_macro_streaming.py --label lut-macro-raw-smoke-v1 --lut-exp --raw-lut --length 1024 --heads 1 --cores 1 --sample-rows 1024 --check-preprocess --iters 0
python experiments/sdpa-l2/bfp4-lofi-v2/exp_lut_macro_streaming.py --label lut-macro-fused-smoke-v1 --lut-exp --length 1024 --heads 1 --cores 1 --sample-rows 1024 --check-preprocess --iters 0
```

Require both JSON `output_sha256` values equal. Then compare identical resident
parameters; all256 final query rows are checked against the original BF16 FP64
reference, using repeated-K/V equivalence (not distinct262K keys).

```bash
python experiments/sdpa-l2/bfp4-lofi-v2/exp_lut_macro_resident.py --label lut-macro-raw-resident-v1 --lut-exp --raw-lut --q-repeats 16 --k-chunks 512 --warmup 3 --iters 5
python experiments/sdpa-l2/bfp4-lofi-v2/exp_lut_macro_resident.py --label lut-macro-fused-resident-v1 --lut-exp --q-repeats 16 --k-chunks 512 --warmup 3 --iters 5
```

Also require resident raw/macro output hashes equal. All three controls emit
`exp_refiner`, exact coefficient bits, source hashes, finite/L2 gates, unchanged
CB audit and trace-repeat equality. Fullchip32K/H10/110-core timing follows only
after small and resident equality, and reports useful FLOPs excluding SFPU work.

## Primary implementation references

- [LOADMACRO scheduling, register override, captured stores and collision rules](https://github.com/tenstorrent/tt-isa-documentation/blob/main/BlackholeA0/TensixTile/TensixCoprocessor/SFPLOADMACRO.md).
- [SFPLUTFP32 TABLE1 and fixed L3 input](https://github.com/tenstorrent/tt-isa-documentation/blob/main/BlackholeA0/TensixTile/TensixCoprocessor/SFPLUTFP32.md).
- [MAD latency and absence of automatic macro stalls](https://github.com/tenstorrent/tt-isa-documentation/blob/main/BlackholeA0/TensixTile/TensixCoprocessor/SFPMAD.md).
- Frozen `hybrid-mixed-v1/candidate/.../ckernel_sfpu_sdpa.h`:
  `init_sdpa_refine_loadmacros`, `restore_sdpa_grid_macro_instructions` and
  `calculate_sdpa_exp_refine_loadmacro` (lines362–435).
- Current Blackhole `ckernel_sfpu_exp.h` native unclamped init (lines935–1048).

Python syntax, source existence and unchanged CB/compute config are checked
locally; JIT/device results must be added by the parent before acceptance.
