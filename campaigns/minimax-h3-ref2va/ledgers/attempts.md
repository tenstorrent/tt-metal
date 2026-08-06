# Attempts — MiniMax-H3 `ref2va`

One row per round, win or loss. Answers "what has been tried?"

`| round | date | sha | scope | hypothesis | metric | quality | verdict |`

| r | date | sha | scope | hypothesis | metric | quality | verdict |
|---|---|---|---|---|---|---|---|
| r1 | 2026-08-06 | PENDING | host-packing | Port `build_ref2va_packed_sequence` + `build_ref2va_presentation` + media prep + the sequential rotary span | 61 host tests, no device | bit-exact vs reference (5 request shapes) | kept |
| r2 | 2026-08-06 | PENDING | reference-preparation | `references.py`: prepare at own resolution, audio hop pad, typed condition-block split | 24 host tests, no device | bit-exact vs `MiniMaxH3Ref2VASetupStep` (7 media kinds) | kept |
| r3 | 2026-08-06 | PENDING | transformer-condition-stream | `condition_blocks: [(tensor, modality)]` replacing `condition_1BKC`; audio blocks take `audio_proj_in` | 2-layer, 4x8, 9 cases | 12/12 existing PCCs IDENTICAL; interleaved 99.9974/99.9975 | kept |
