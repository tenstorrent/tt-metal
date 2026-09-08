# work log — functional decoder

- 2026-09-08 19:52 first device run: opening the full (1,4) parent mesh and a (1,1) submesh hung on the 800 MB
  embedding upload (`TT_THROW: TIMEOUT ... device unrecoverable`, device 1). Reset with `tt-smi -r`. Fix: true
  single-device mesh via `TT_METAL_VISIBLE_DEVICES=0`. Recorded as infrastructure, not a model result.
- 20:03 segfault reading the tapped hidden state: `LMHead.forward` deallocates its input -> tap now clones for the head.
- 20:16 full-model teacher-forced test showed PCC ~0 -> localised with `test_slow_layer.py`: layers/tail/Generator path
  all >= 0.9987; the bug was the test's golden layout (`frames.pt` = generated frames only). Fixed.
- Weight cache: `~/.cache/fish_s2_pro/tt_cache/P150/tensor_cache_bfp8` (5.6 GB, 410 files); warm load < 30 s.
- 20:34 teacher-forced vs fp32 CPU golden (short/no_ref, 68 frames), 1 chip: bfp8 weights -> cb0 top-1 0.838, top-5 1.00,
  logits PCC 0.9987, hidden PCC 0.9991; bf16 weights -> cb0 top-1 0.809 (not better). The semantic argmax over 4096
  near-tied candidates is limited by bf16 activation math, not weight quantization; thresholds are tied to the
  bf16-vs-fp32 CPU floor (stage 01 tf_floors.json). Stage 04 (TP2/TP4) started.
- 20:48 stage 03 PASSED (1 chip, bfp8): short/no_ref, short/ref, medium/no_ref all logits PCC >= 0.9985, hidden >= 0.9986,
  cb0 top-5 1.00, cb0 top-1 0.84/0.90/0.84 (advisory until the CPU bf16 floor is in). Root cause of the earlier
  garbage: the LM-head tap kept ONE stash, so after a decode trace existed a later prefill made decode replays read a
  stale tensor -> per-mode stashes (prefill/decode).
- 20:54 stage 04: TP2 (P300, FABRIC_1D) and TP4 (P150x4 view of the QB2) both reproduce the 1-chip outputs at
  PCC >= 0.998 (logits and hidden), same top-1 within argmax noise. Free-running smoke: slow step ~34 ms (1 chip),
  27 ms (TP2), 30 ms (TP4); the CPU torch fast decoder costs ~300 ms/frame -> Phase B (fast decoder on TTNN) is
  the dominant performance item. Weight-cache warm loads: 1x1 < 30 s, 1x2/1x4 similar once converted.
- 21:20 stage 05 (Phase A end-to-end, TT slow + CPU fast + CPU codec): 16/16 clips generated on 1x1 and 1x4, all stop
  on <|im_end|>; short/no_ref greedy = 68 frames (golden 67), whisper-tiny WER 0.0 ("The quick brown fox jumps over
  the lazy dog."), RMS 0.39, no repeated frames. RTF ~8.4: decode 22.7 s of which CPU fast decoder 20.1 s, slow
  tower 2.6 s (38 ms/frame), codec 2.0 s. Gate failed only because gates.json was not calibrated yet.
- 21:21 Phase B first run: TTNN fast decoder (Attention1D/MLP1D/RMSNorm1D/RotarySetup1D/Embedding1D/LMHead1D, two traces)
  vs torch fp32 on 8 real hidden states: logits PCC min 0.9995, argmax agreement 0.847, 48 ms/frame (10 steps) vs
  ~300 ms/frame on CPU; build+warmup 12 s. Expected frame budget now ~38 ms slow + 48 ms fast => ~11.6 frames/s (RTF ~1.9).
- 21:27 stage 06 PASSED: Phase B end-to-end (TT slow + TT fast + CPU codec) short prompt: 68 frames / 3.16 s audio,
  RTF 3.3-3.5 (per frame: slow 38 ms, fast 52 ms incl. 9 host logits readbacks + 10 host writes, codec 26 ms;
  prefill fixed cost 1.8 s). Phase D targets: on-device sampling in the fast trace, traced prefill, codec on TT.
