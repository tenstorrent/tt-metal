# Pytest call65 cap4096 extension

Candidate only: not admitted and not run. This standalone full-precision case takes the exact full-mode Pytest call65 seed44 cap2048 fixture, keeps its full prompt IDs, continuation metadata, sampling, and seed, and requests 4096 free-run tokens instead of 2048. The prior cap2048 output produced 2047 decode replays while writing distinct serialization code; this is a progressing-cap extension, not a confirmed repeated anomaly.

The estimate scales the measured 2047-replay decode time by cumulative live sequence context to the cap-safe 4095 replays, preserves measured prefill/host overhead, and adds the actual same-mode T model load and first trace-preparation costs. The planning range is 0.7–1.7× variable work with fixed load/preparation held constant. No device work or score change is authorized by this candidate.
