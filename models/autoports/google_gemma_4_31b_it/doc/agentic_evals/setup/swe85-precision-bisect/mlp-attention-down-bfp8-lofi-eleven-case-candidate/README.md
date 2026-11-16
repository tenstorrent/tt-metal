# P8 MLP BFP8-LoFi eleven-case candidate (inert)

Host-prepared, unadmitted, and unexecuted full-mode diagnostic. Relative to P6, only MLP gate/up weights change from BFP4/LoFi to BFP8-LoFi/LoFi. Attention, down projection, and lm_head remain BFP8-LoFi/LoFi; all other policy fields are unchanged. This is not an all-BF16 arm.

Cases use exact P6 fixed108 manifest indices 0, 32, 33, 58, 66, 67, 68, 74, 85, 93, and 96, preserving fixtures, seeds, and caps. P7 index 0 returned a 634-token EOS-terminated result; index 32 emitted malformed tool opener ID49 and stopped naturally at 16 tokens. P8 tests a distinct MLP-only hypothesis relative to P6; no repair is assumed.

`forecast.json` derives each case timing from the exact P6 result JSON selected by the manifest row, plus one measured P6 model load and P6 completion overhead. The corrected point scenario is 556.7 s with a broad 500–1800 s planning range; this is not a P8 measurement. No admission or launch record is present.
