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
