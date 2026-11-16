# P7 attention HiFi2 ten-case candidate (inert)

This is a host-prepared, unadmitted and unexecuted ten-case full-mode diagnostic. It holds P6 fixed except attention precision: BFP8/LoFi becomes BFP8/HiFi2. In `precision.py`, `bfp8` maps to `bfloat8_b`/HiFi2; all other policy fields remain equal to P6.

Cases are exact fixed108 manifest indices 0, 32, 33, 58, 66, 67, 68, 74, 85, and 93, preserving each fixture, seed, and cap. Index 85 adds Sympy13551 call1 seed43 cap512, where P6 emitted a potential protocol regression at the tool-call opener (ID49 versus P3 ID48). This is a diagnostic inclusion, not a claim that P7 corrects it.

`forecast.json` is a planning proxy with a 598.9 s point scenario and 455–1250 s envelope, based on the corresponding P6 rows plus one P6 model load. There is no P7 runtime evidence. `command.json` is inert; there is no admission or launch record.
