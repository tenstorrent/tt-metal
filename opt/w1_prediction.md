# W1 traced prediction — LOCKED BEFORE THE RUN

## Call-site census (source-verified, not assumed)
6 gated attentions/block; ALL 6 carry `to_gate_logits` weights in the distilled ckpt
(608 gate keys; 48 blocks x 6 modules = 288, + connectors). So the gate is LIVE on all 6
=> `_gate_is_live()` true => dedup fires on all 6. 48 blocks.

The gather is of `spatial_1BND` = the QUERY-side activation, so its size is set by
`query_input_dim`, NOT by the attention's `dim`:

| attn                  | spatial_1BND | query_input_dim | gather scale |
|-----------------------|--------------|-----------------|--------------|
| attn1 (video self)    | video_normed | 4096            | VIDEO (big)  |
| attn2 (video cross)   | video_ca_in  | 4096            | VIDEO (big)  |
| audio_to_video (a2v)  | video_q_a2v  | 4096            | VIDEO (big)  |
| audio_attn1 (self)    | audio_normed | 2048            | audio (tiny) |
| audio_attn2 (cross)   | audio_ca_in  | 2048            | audio (tiny) |
| video_to_audio (v2a)  | audio_q_v2a  | 2048            | audio (tiny) |

=> **3 BIG + 3 tiny duplicated gathers per block.** (a2v is the trap: it is named "audio_to_
video" and has dim=audio_dim, but its QUERY is video => its gather is VIDEO-scale.)

Shapes: S1 vN=9728 -> 1216 rows/dev at SP=8; S2 vN=38912 -> 4864 rows/dev; aN=256 -> 32 rows/dev.

## Pricing (opt/ccl_census.log — TRACE-priced by slope, not eager)
Video S1: agmm_gate 166.33 | agmm_qkv 325.98 | ag_activation 105.37 | mm_gate 57.94 | mm_qkv 251.25
Audio   : agmm_gate  38.02 | agmm_qkv  60.80 | ag_activation  12.34 | mm_gate 15.57 | mm_qkv  41.51

SELF-attn saving is EXACT (every term measured):
  video: (166.33+325.98) - (105.37+57.94+251.25) = 492.31 - 414.56 = **77.75 us**
  audio: ( 38.02+ 60.80) - ( 12.34+15.57+ 41.51) =  98.82 -  69.42 = **29.40 us**

Decomposition: saving = (G_gate_exposed - AG_standalone) + G_q_exposed
  G_gate_exposed = agmm_gate - mm_gate = 108.39 (video) / 22.45 (audio)  <- gate hides ~0% of its
    gather (108.39 vs standalone AG 105.37) => confirms job 060.
  G_qkv_exposed  = agmm_qkv  - mm_qkv  =  74.73 (video) / 19.29 (audio)  <- QKV hides ~29%.
  check video self: (108.39-105.37) + 74.73 = 77.75 OK ; audio: (22.45-12.34)+19.29 = 29.40 OK

CROSS-attn uses to_q (less compute than to_qkv => hides LESS => saves MORE). Bounded on both
sides by measured terms: G_q_exposed in [G_qkv_exposed, G_gate_exposed].
  video cross saving in [77.75, 111.41], mid ~94.6 us
  audio cross saving in [29.40,  32.56], mid ~31.0 us

## PREDICTED dSTEP_MS
S1 per block = 77.75 + 94.6 + 94.6 (video) + 29.40 + 31.0 + 31.0 (audio) = ~358 us
  x48 blocks => **S1 dSTEP_MS = -17 ms (range -15 to -21), 348.3 -> ~331 ms, -4.9%**

S2: video AG scales with bytes. Fit AG(b) = 11.1 + 3.786e-5*b from the two measured AGs
  => video S2 AG (4864 rows) ~ 388 us. Holding the exposed fractions:
  S2 per block ~ 286 + 337 + 337 (video) + 91 (audio) = ~1051 us
  x48 => **S2 dSTEP_MS = -50 ms (range -44 to -62), 1092.5 -> ~1042 ms, -4.6%**
  (S2 is the softer number: linear-AG extrapolation + assumed hiding fraction.)

## The discriminator
Eager measured -1.54 ms/BLOCK. Device model accounts for only -0.358 ms/block.
=> ~77% of the eager win is HOST-DISPATCH and will NOT ship; ~23% is real device time.

- Land at ~-17 ms (S1): device model CONFIRMED -> **W1 SHIPS** (real, but ~4x smaller than eager sold).
- Land at ~0 (within a few sigma of 348.3): even the trace-priced census does not transfer to the
  full model -> **W1 DEAD**, eager mirage, retire it. sigma ~0.1-0.3 ms, so -17 ms would be >50 sigma.
  There is no ambiguous middle at this precision.
