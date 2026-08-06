# Attempts — MiniMax-H3 `ref2va`

One row per round, win or loss. Answers "what has been tried?"

`| round | date | sha | scope | hypothesis | metric | quality | verdict |`

| r | date | sha | scope | hypothesis | metric | quality | verdict |
|---|---|---|---|---|---|---|---|
| r1 | 2026-08-06 | 5c8adce1e85 | host-packing | Port `build_ref2va_packed_sequence` + `build_ref2va_presentation` + media prep + the sequential rotary span | 61 host tests, no device | bit-exact vs reference (5 request shapes) | kept |
| r2 | 2026-08-06 | 5ec8933bbfa | reference-preparation | `references.py`: prepare at own resolution, audio hop pad, typed condition-block split | 24 host tests, no device | bit-exact vs `MiniMaxH3Ref2VASetupStep` (7 media kinds) | kept |
| r3 | 2026-08-06 | 5ec8933bbfa | transformer-condition-stream | `condition_blocks: [(tensor, modality)]` replacing `condition_1BKC`; audio blocks take `audio_proj_in` | 2-layer, 4x8, 9 cases | 12/12 existing PCCs IDENTICAL; interleaved 99.9974/99.9975 | kept |
| r4 | 2026-08-06 | 5ec8933bbfa | shape-probe | do 46080/81664/111616 fit at full depth against `transformer_ref` | warm 2.11/3.26/5.45 s | 6 passed, outputs finite and non-degenerate | kept |
| r5 | 2026-08-06 | 5ec8933bbfa | reference-encode | per-modality device encode vs the reference encoder step, on real media | 3 passed, 746 s | image 99.9905% / audio 99.9910% / video 99.9927% | kept |
| r6 | 2026-08-06 | 4d04f289379 | e2e | ref2va end to end at 46080, plus the t2va/fl2va no-regression gate | 210.7 s compute; regression 5 passed | conditioning signal 0.080038 vs floor 0.000000; t2va metrics bit-identical | kept |
| r7 | 2026-08-06 | db76ad3807e | adaln-fourth-level | add the audio-conditioning timestep as a fourth AdaLN level, for ref2va only | 270.5 s / 372.0 s compute | discriminator + order gates pass; seam check red on one case | kept |
| r8 | 2026-08-06 | 440bd05c058 | seam-bar-and-consolidation | separate seam bars per axis on measured evidence; confirm the whole ref2va file green in one process | 5 passed, 1435 s | 8/8 gates green | kept |
| r9 | 2026-08-06 | e10e6dda34e | quality-bars | record CLIP + VBench on all three shapes, then set bars from the measurements | 3 passed, 1383 s | 3 of t2va's 6 bars would fail, none a defect | kept |
| r10 | 2026-08-06 | e10e6dda34e | warm-perf-baseline | warm ref2va latency with the padded_len validity gate | 73.6 / 193.3 / 216.1 s | 3 passed | kept |
| r11 | 2026-08-06 | PENDING | hoist-invariant-upload | hoist the provably-redundant per-step conditioning upload | -0.8 / -0.1 / -0.3 % | inside the +-8 % noise floor: NOT a win | kept, forensic |
