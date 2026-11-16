# P7 attention HiFi2 candidate (inert)

This is a host-prepared, unadmitted and unexecuted nine-case full-mode diagnostic. It holds P6 fixed except attention precision: BFP8/LoFi becomes BFP8/HiFi2. In `precision.py`, `bfp8` maps to `bfloat8_b`/HiFi2; all other policy fields stay equal to P6.

Cases are exact fixed108 manifest indices 0, 32, 33, 58, 66, 67, 68, 74, 93, preserving prompt fixture, seed and cap. Indices 0 and 32 test the two P6-only findings; the other seven are prior targets.

`forecast.json` is a broad planning proxy (450–1200 s), based on measured P6 rows and one model load. No P7 runtime evidence exists. `command.json` is inert; there is no admission or launch record.
