# Bring-up LOG (append-only; see models/common/bringup_kit/README.md)

## E261001-0900-a01
- date: 2026-10-01T09:00:00
- commit: c37eb3b994e
- branch: ssinghal/dsv4p1
- host: 44
- track: correctness
- status: verified
- claim: 40 layers, ISL 4096, B=16, no env: first-token argmax 16/16 vs reference, first-token PCC 0.99180
- evidence: models/demos/blackhole/deepseek_v41_flash/PREFILL_AUTO_NOTES.md
- paths: models/demos/blackhole/deepseek_v41_flash/tt
- data: {"agree": 16, "n": 16, "pcc": 0.9918}

## E261001-0910-a02
- date: 2026-10-01T09:10:00
- commit: c37eb3b994e
- branch: ssinghal/dsv4p1
- host: 44
- track: correctness
- status: verified
- claim: gsm8k_b64 demo session, 40 layers: 64/64 correct vs gold
- evidence: models/demos/blackhole/deepseek_v41_flash/PREFILL_AUTO_NOTES.md
- paths: models/demos/blackhole/deepseek_v41_flash/tt, models/demos/blackhole/deepseek_v41_flash/demo
- data: {"n": 64, "correct": 64}

## E261002-1000-a03
- date: 2026-10-02T10:00:00
- commit: 5a15bdd2ccd
- branch: ssinghal/dsv4p1
- host: 44
- track: correctness
- status: verified
- claim: G3 layers 0-20 ISL 2048 chunked sparse prefill: per-layer decode-out PCC min 0.992 (layer 10), max 0.99999
- evidence: models/demos/blackhole/deepseek_v41_flash/INTEGRATION_NOTES.md
- paths: models/demos/blackhole/deepseek_v41_flash/tt
- data: {"pcc": 0.992}

## E261003-1100-a04
- date: 2026-10-03T11:00:00
- commit: cd8b7c7dd48
- branch: ssinghal/dsv4p1
- host: 44
- track: determinism
- status: inferred
- claim: bfp8/bfp4 decode repeats are bit-identical in PCC (notes say 'bit-identical in PCC between repeats', outputs not compared bitwise)
- evidence: models/demos/blackhole/deepseek_v41_flash/BFP4_NOTES.md
- paths: models/demos/blackhole/deepseek_v41_flash/tt

## E261006-1043-a05
- date: 2026-10-06T10:43:13
- commit: a9593c4477a
- branch: ssinghal/dsv4p1
- host: 45
- track: prefill
- status: verified
- claim: isl4k_b16, 40 layers: TTFT 9358 ms, prefill 6360 tok/s; decode 42.7 ms/token = 23.4 tok/s/user
- evidence: /mnt/tt-data/ssinghal/dsv4-logs/grid_b16_G4k.log
- paths: models/demos/blackhole/deepseek_v41_flash/tt, models/demos/blackhole/deepseek_v41_flash/demo
- supersedes: E261006-1100-a06
- data: {"ttft_ms": 9358, "prefill_tok_s": 6360, "decode_ms_tok": 42.7, "tok_s_user": 23.4}

## E261006-1100-a06
- date: 2026-10-06T11:00:00
- commit: a9593c4477a
- branch: ssinghal/dsv4p1
- host: 45
- track: prefill
- status: verified
- claim: (older, wrong) isl4k_b16 TTFT 24900 ms
- evidence: /mnt/tt-data/ssinghal/dsv4-logs/grid_b16_G4k.log
- paths: models/demos/blackhole/deepseek_v41_flash/tt
- data: {"ttft_ms": 24900, "prefill_tok_s": 2391}
