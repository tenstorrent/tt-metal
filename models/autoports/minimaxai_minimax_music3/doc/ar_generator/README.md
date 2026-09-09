# Stage 04 — autoregressive generator (global LLM + depth decoder + sampling)

Work log for the AR stage of MiniMax-Music3 on one Blackhole chip: diffusers' `MiniMaxMusic3TokenizeStep`
+ `MiniMaxMusic3AutoregressiveStep` driving the stage-02 `MusicLLM` and the stage-03 `DepthDecoder`.

* implementation: [`../../tt/ar_generator.py`](../../tt/ar_generator.py) (`ARGenerator`, `sample_top_k`),
  [`../../tt/prompt.py`](../../tt/prompt.py) (`clean_caption`, `normalize_lyrics`, `assemble_prompt`, `PromptEncoder`)
* changes to earlier stages: [`../../tt/llm.py`](../../tt/llm.py) gained `MusicLLM(logits_window=...)` /
  `set_logits_window`, `prepare_decode_inputs`, `decode_windowed` (the stage-02 API and tests are unchanged;
  `DepthDecoder` / `DepthStepTrace` are used as delivered by stage 03)
* tests: [`../../tests/test_ar_generator.py`](../../tests/test_ar_generator.py) (gate, 5 tests, ~6 min on device incl. the 3-min end-token run)
* scripts: [`../../scripts/end_token_probe.py`](../../scripts/end_token_probe.py) (device: long free run with
  end-token statistics), [`../../scripts/end_token_control_cpu.py`](../../scripts/end_token_control_cpu.py)
  (fp32 CPU reference control of the same run, diffusers venv)
* measured numbers: [`pcc/results.json`](pcc/results.json) (written by the gate tests),
  [`pcc/end_token_probe_7.json`](pcc/end_token_probe_7.json), [`pcc/end_token_control_cpu_7.json`](pcc/end_token_control_cpu_7.json)
* local-only (gitignored): `generated/ar_full.log`, `generated/gate04.log`, `generated/end_token_*.log`

Hardware: one chip of the P300x2 host (`TT_METAL_VISIBLE_DEVICES=0`, board id `000004613193411b`, reported as
P150, `tt-smi -s`), 1x1 mesh, trace region 90 MB. Software: ttnn from `~/tt-metal` at `e946955cc15`
(shared prebuilt binary), model code in worktree branch `jashan/minimax-music3`; stage 02 at `7e504d4acc2`,
stage 03 at `21a5e24ff3b`; golden reference = stage-01 fp32 diffusers run (`~/mm3-bringup/reference`,
seed 7, 10 s clip, 250 frames).

## What was built

```python
ar = ARGenerator(llm, depth, tokenizer_dir=f"{MM3_WEIGHTS}/tokenizer")
text_ids = ar.build_text_ids(prompt, lyrics)                      # [2, L] int64, row 1 = CFG prompt
out = ar.generate(prompt, lyrics, max_frames=250, seed=7)         # free running
out = ar.generate(prompt, lyrics, max_frames=250, seed=7,
                  teacher_codes=codes, teacher_frame0_codes=c0)  # teacher forced
out["frame_hiddens"]  # [1, F, 32768] fp32   out["codes"]  # [F, 8]   out["frames"]  # F
out["stopped_by"]     # "end_token" | "max_frames" | "teacher_codes" | "context"
out["timings"]        # host wall seconds: prefill, llm_step, depth, host, per_frame[]
```

`build_text_ids` is a verbatim transcription of `_clean_caption`, `_normalize_lyrics` and the prompt template
from diffusers `encoders.py`; the tokenizer is `transformers.Qwen2Tokenizer` (5.15.0) from the checkpoint's
`tokenizer/` directory, the >5000-token `ValueError` and the CFG row (`ids[:, 1:-2] = 151654`) are the
reference's.

`generate` transcribes `MiniMaxMusic3AutoregressiveStep.__call__` + `_generate_depth_codes` + `_sample_top_k`
+ `_embed_audio_frame`:

| reference step | here |
|---|---|
| `embed_tokens(text_ids)` + prefill | host gather from the bf16 embedding table, `MusicLLM.prefill` (both rows) |
| `lm_head(last_hidden)`, `vocab_mask`, CFG 1.5, conditional top-50 restriction, re-mask | the decode trace returns the tile-aligned column window `[151648, 168064)` untilized; the host puts the logical window `[151670, 168059)` (end token + 16384 semantic codes) back into a `-inf`-filled `[2, 200000]` tensor and runs the reference arithmetic unchanged |
| `_sample_top_k(guided, generator)` | identical code on the full-length tensor with a CPU `torch.Generator(seed)` (the multinomial draws one random number per category, so the tensor length is part of the sampling contract) |
| end token -> break; frame 0 not emitted | same |
| `_generate_depth_codes`: projection of `last_hidden` and `embed(sem + offset)`, 7 steps, per-step CFG 1.5 + top-50, `code.repeat(2)`, `hidden[:1]` collected | `DepthStepTrace.begin_frame(hidden_device_tensor, sem_embed)` then 7 `step()` replays; per step the host reads the `[2, 1024]` head logits and the `[1, 4096]` row-0 hidden |
| `frame_hiddens.append(cat(last_hidden[:1], depth_hidden))` | same, fp32 on host |
| `_embed_audio_frame` feedback | host `reference/hf_llm.embed_audio_frame` (bf16 exactly as HF: `embed + residual_sum` in bf16, then `* 8**-0.5`), written into the persistent decode input |
| `max_frames = min(duration * 25, 9000)` | `max_frames` is the argument, capped at 9000; additionally stops with `"context"` if `L + frame_index` would reach 10240 |

Per frame the device allocates nothing. Allocation / capture order in `ARGenerator.__init__`: `MusicLLM.prepare_decode_inputs()`
(persistent decode input / position / RoPE-index tensors), then `DepthStepTrace(depth)` (its persistent buffers and
both depth traces), then the backbone trace is captured lazily on the first decode after the prefill. Every buffer
that lives across a trace replay therefore exists before that trace is captured (depth buffers before the backbone
capture, backbone inputs before the depth captures); the backbone trace outputs are consumed (hidden copied into the
depth seed buffer, window read back) before the depth replays of the same frame. Steady-state host work per frame:
one 256 KB input write, one trace replay, two read-backs (256 KB hidden, 1 MB window), 7 depth replays with 3 small
buffer writes and 2 read-backs each, and the sampling. `decode_stats` over a 50-frame run: 50 replays, 50 input
refreshes, 1 position refresh (after the prefill), 0 trace re-captures.

## Evidence

All numbers below come from `pcc/results.json` written by the gate run on 2026-09-09 (commands in "How to run").

### Prompt contract (host only)

| check | result |
|---|---|
| `build_text_ids(golden prompt, lyrics)` vs `text_ids.pt` | exact match, shape `[2, 104]`; CFG row = `<|im_start|>`, 101 x 151654, `<|im_end|>`, `<|audio_start|>` |
| lyrics tags on the text line (`"[Verse] Morning light"`, `"[chorus][bridge] la la"`, `"plain line [intro] inline"`) | normalizes to `[start]\n[verse]\n[chorus][bridge]\nplain line\n[intro]\ninline` (checked against diffusers' `_normalize_lyrics` in the ref venv) |
| markdown caption (`# `, `- `, `**bold**`, `*em*`, `---`, `<|key C major|>`, `• `, 4-space runs) | cleaned exactly like `_clean_caption`; tokenizes to the same ids as the plain caption |
| 5000-token prompt | tokenizes (`[2, 5000]`); 5001 raises `ValueError("... maximum is 5000")`; empty prompt / blank lyrics raise |

### Teacher-forced run over the golden codes (`test_teacher_forced_frame_hiddens_vs_golden`)

Prompt = golden, `teacher_codes = sampled_codes.pt` (250 emitted frames), `teacher_frame0_codes` = group 0 of
`sampled_raw.pt` (the un-emitted frame 0, whose feedback shapes every later state; `sampled_codes.pt` does not
contain it - see decisions). The returned `codes` / `frame0_codes` equal the inputs.

| quantity | value | bar |
|---|---|---|
| `frame_hiddens` PCC vs `frame_hiddens.pt`, overall (`[1, 250, 32768]`) | **0.99941** | >= 0.99 |
| per-frame PCC: min / mean | **0.99512** (frame 132) / 0.99941 | min >= 0.99 |
| per-frame backbone part (cols 0..4095): min | 0.99176 | (diagnostic) |
| per-frame depth part (cols 4096..32767): min | 0.99647 | (diagnostic) |
| golden semantic code = argmax of the device's guided distribution | 42.6 % of the 251 frames | (diagnostic) |
| golden semantic code inside the device conditional row's top-50 | 99.2 % (249 / 251) | (diagnostic) |
| wall for 250 frames (idle host) | 17.9 s (prefill 0.17 s, LLM steps 9.7 s, depth 7.9 s, host 0.1 s) | — |

Per-frame PCC curve (every 25th frame; the full 250-entry list is `per_frame_pcc` in `results.json`):

| frame | 0 | 25 | 50 | 75 | 100 | 125 | 150 | 175 | 200 | 225 |
|---|---|---|---|---|---|---|---|---|---|---|
| PCC | 0.99940 | 0.99949 | 0.99934 | 0.99967 | 0.99961 | 0.99959 | 0.99938 | 0.99958 | 0.99967 | 0.99923 |
| backbone / depth | 0.99911 / 0.99967 | 0.99931 / 0.99965 | 0.99891 / 0.99961 | 0.99965 / 0.99971 | 0.99948 / 0.99971 | 0.99955 / 0.99964 | 0.99914 / 0.99949 | 0.99946 / 0.99971 | 0.99959 / 0.99974 | 0.99908 / 0.99934 |

No drift over the 250 positions (104..354): the curve is flat within 0.995-0.9997. The 42.6 % top-1 agreement
is the expected kind of value for a top-50 *sampled* golden (the golden code is itself a draw, not the argmax); the
99.2 % in-top-50 rate is the number that matters for the CFG restriction, matching stage 02's `golden_code_rank_check`.

### Free-running generation (`test_free_running_generation`, 50 frames, golden prompt)

| check | result |
|---|---|
| seed 7 twice | identical `codes`, `frame0_codes` and bit-identical `frame_hiddens` |
| seed 11 | different codes |
| code ranges | semantic in `[0, 16384)`, residual in `[0, 1024)`, all hiddens finite |
| semantic-code distribution, seed 7 | 33 distinct codes in 50 frames, most common (2782) 5 x = 10 % (bar: <= 30 %) |
| position refreshes across a run | 1 (after prefill); 50 replays / 50 input refreshes / 0 re-captures |

### Performance (warmed free-running loop, host wall, frames 5..49 of the seed-11 run)

| section | ms / frame | share |
|---|---|---|
| backbone step (`decode_windowed`: 256 KB input write, trace replay, 256 KB + 1 MB read-backs) | 37.5 | 50 % |
| depth loop (seed replay + 7 step replays + 7 x (458 KB + 256 KB) read-backs) | 33.5 | 44 % |
| host (CFG + top-50 sampling over 200k, 7 x top-50 over 1024, feedback embedding, bookkeeping) | 4.3 | 6 % |
| **total** | **75.4 ms / frame = 13.3 frames/s** | |
| prefill of the 104-token golden prompt (both rows) | 137 ms | |

Realtime is 25 frames/s (40 ms / frame): the loop is **1.9x slower than realtime** (a 10 s clip's 250 frames
take about 19 s of AR time; the whole golden CPU fp32 pipeline run - AR + DiT + vocoder - took 1076 s, `manifest.json`). Both device sections match their
stage-02 / stage-03 measurements (traced backbone step 37.1 ms incl. positions on device; depth frame 31.1 ms
traced), i.e. the generator adds 3-4 ms of host work per frame and nothing else. The first full-file run
(`generated/ar_full.log`, idle host) measured 13.7 frames/s (73.1 ms = 37.0 + 33.0 + 3.3); the numbers above are
the recorded `results.json` run. **Host contention matters**: with the fp32 CPU control (10 torch threads) running
on the same host the identical test measured 9.3 frames/s (107.8 ms = 43.2 + 42.3 + 22.7 host) - the read-back /
dispatch path and the sampling are host-thread-sensitive, so perf numbers must be taken on an idle host (the
`results.json` run was taken with the control process SIGSTOPped).

The gap to realtime is the two DRAM-bound device loops (backbone ~10 GB of weights per step at bf16 attention /
bfp8 MLP; depth 1.14 GB per step x 7): the levers are the stage-07 dtype sweep (bfp8 attention weights / KV
cache), overlapping the depth loop of frame *f* with the backbone step of frame *f+1* (they are independent once
the frame's codes are known - the backbone step only needs the feedback embedding, the depth hiddens are only
appended to the output), and moving the sampling on device.

### Qualitative: end token and code distribution

The stage asks that the semantic-code distribution is not degenerate and that the end token eventually appears
for a short lyric with a large `max_frames`. Prompt for the short-lyric run: caption "Genre: acoustic pop. BPM: 96.
Key: C major. A short intimate vocal phrase over one guitar.", lyrics `[verse]\nMorning light through the pine`,
seed 7.

| run | frames | stop | distinct semantic codes | most common code |
|---|---|---|---|---|
| golden prompt, free running, 50 frames (gate) | 50 | `max_frames` | 33 | 10 % |
| short lyric, first attempt with a 1500-frame (60 s) cap (`generated/ar_full.log`, not kept as a test) | 1500 | `max_frames` | 859 | 1.1 % |
| short lyric, `scripts/end_token_probe.py`, 9000-frame cap (`pcc/end_token_probe_7.json`) | **2782 = 111.3 s of audio** | **`end_token`** | 1331 | 1.0 % |
| short lyric, gate test `test_end_token_for_short_lyric` (4500-frame cap) | 2782 | `end_token` | 1331 | 1.0 % |

So the end token does appear by itself; a one-line lyric still yields a ~2-minute song, which is why the first
60 s cap was too short (decision 6). The probe also logged, per frame, the end token's rank in the conditional
row, its logit gap to the row maximum and its probability under the final top-50 sampling distribution:

| frames | end-token rank in conditional row (min / median) | logit gap to max (min / median) | frames with sampling prob > 0 |
|---|---|---|---|
| 0-499 | 1079 / 5878 | 6.7 / 24.2 | 0 |
| 500-999 | 1126 / 5313 | 12.9 / 25.3 | 0 |
| 1000-1499 | 1737 / 6067 | 14.7 / 27.7 | 0 |
| 1500-1999 | 1971 / 6278 | 15.7 / 27.9 | 0 |
| 2000-2499 | 1820 / 6683 | 15.1 / 27.9 | 0 |
| 2500-2783 | 0 / 4501 | 0.0 / 26.1 | 2 |

The ending is a switch, not a drift: at step 2782 the end token jumps to rank 11 (gap 9.6) and at step 2783 to
rank 0 with sampling probability 1.0 (the whole top-50 mass), i.e. the backbone signals "song over" decisively.
Until then the end token never enters the sampling set (its probability is exactly 0 on every frame), so there is
no premature-ending risk from the bf16 device logits either. The fp32 CPU control of the same run
(`scripts/end_token_control_cpu.py`, 4000-frame cap, about 3 s / frame on 10 cores) is recorded in
`pcc/end_token_control_cpu_7.json`; see the note at the end of this section for how far it got.

The probe's `collect_end_token_stats=True` costs about 49 ms of host time per frame (two top-k passes over the
200k-vocabulary tensor); it is off in `generate` by default and in every timed run.

CPU_CONTROL_NOTE

## How to run

```bash
source ~/mm3-bringup/common.sh && cd $MM3_WT
# gate (5 tests, ~6 min incl. model build from the converted-weight cache and the 2782-frame end-token run)
with_hw_lock timeout 5400 $MM3_PY -m pytest $MM3_MODEL_DIR/tests/test_ar_generator.py -m "not slow" -x -q -p no:cacheprovider
~/mm3-bringup/checks/04.sh
# long free run with end-token statistics (device) and the fp32 CPU control (no device, hours)
with_hw_lock timeout 3600 $MM3_PY $MM3_MODEL_DIR/scripts/end_token_probe.py --max-frames 9000 --seed 7
$MM3_REF_PY $MM3_MODEL_DIR/scripts/end_token_control_cpu.py --max-frames 2500 --seed 7
```

## Decisions taken (nobody to ask)

1. **Sampling and the feedback embedding on the host.** The stage prompt allows host sampling; the reference
   samples with `torch.multinomial` on a CPU generator, so keeping it on the host makes the device path a
   drop-in for the diffusers loop (same generator contract, same tensor length). The feedback embedding is
   two table gathers of 4096 values; doing it on the host from the bf16 tables (1.6 GB `embed_tokens` +
   58 MB `audio_embeddings`, loaded once) reproduces HF's bf16 arithmetic exactly and avoids allocating device
   tensors after the traces exist. Cost: 3.3 ms / frame (5 %). On-device sampling / feedback is a stage-05+
   optimization item, not a correctness one.
2. **Logits window on device.** Reading 200k logits per step (12.8 MB) was the stage-02 path; the AR loop only
   needs 16385 columns. The decode trace now slices the tile-aligned window `[151648, 168064)` from the tiled
   logits and untilizes only that (`MusicLLM.set_logits_window`), so the read-back is 1 MB and the full-vocab
   untilize disappears from the trace. The full tiled logits are still returned by `decode(read_back=True)`.
   The stage-02 tests build `MusicLLM` without a window and are unaffected.
3. **Frame 0 codes for teacher forcing.** `sampled_codes.pt` holds the *emitted* frames only, but the loop feeds
   frame 0's codes back before the first emitted frame, so a teacher-forced reproduction needs them. They are
   recovered from `sampled_raw.pt` (every top-k draw; group 0 = frame 0) and passed as `teacher_frame0_codes`;
   the golden fixture asserts groups 1..250 equal `sampled_codes.pt`. Without them the loop samples frame 0 with
   the seed (documented in the docstring).
4. **Trace-lifetime order** (see "What was built"): the backbone decode inputs are allocated explicitly before the
   depth traces are captured (`prepare_decode_inputs`), instead of lazily on the first decode. No
   "Allocating device buffers is unsafe" hazard is introduced by the per-frame loop (it allocates nothing).
5. **Context bound.** `L + frame_index` must stay below `max_seq_len = 10240`; the loop stops with
   `stopped_by = "context"` instead of raising. The reference has no such bound (HF RoPE would extrapolate), but
   the checkpoint advertises 10240 positions and the pipeline's own caps (5000 + 9000) can exceed it.
6. **End-token check cap.** The first version of the test capped the short-lyric run at 1500 frames (60 s) and
   failed: the model had not ended. Rather than weaken the assertion, a 9000-frame probe with per-frame end-token
   statistics was run (it ended at 2782 frames, cleanly) and the gate test now uses a 4500-frame (3 min) cap; it
   adds about 3.5 min to the gate. Generation is deterministic per seed on device, so the count is reproducible.

## Open risks / hand-off

* Throughput is 13.3 frames/s vs 25 realtime; all of it is the two device loops measured in stages 02/03.
  Hand-off to stage 05/07: overlap depth(f) with backbone(f+1), on-device sampling, dtype sweep.
* Long-prompt precision (stage 02: prefill hidden PCC 0.982 at 5000 tokens) is inherited; this stage only
  measured the 104-token golden prompt end to end.
* `frame_hiddens` is bf16-derived (device hidden states); the DiT stage consumes it. PCC 0.9994 vs fp32.
