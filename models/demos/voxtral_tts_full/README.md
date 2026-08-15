# Voxtral-TTS (`voxtral-tts-full`) on Tenstorrent

Text in, 24 kHz speech out, computed on device by the seven graduated TTNN components from
bring-up. One task head (`config.task = "text-to-speech"`; `AutoModel` and `AutoModelForCausalLM`
both map to `VoxtralTtsForConditionalGeneration`), so there is exactly one Call.

## Call 1 — text-to-speech

| | |
|---|---|
| **Input** | text → prompt ids (HF tokenizer, in the layout `voxtral_tokenizer_ref.build_prompt` defines) + a voice preset `[169, 3072]` from `assets/voice_embedding/<voice>.pt` |
| **Output** | 24 kHz mono waveform `[1, 1, T*1920]` (written as a WAV) + the emitted audio-code frames `[T, 37]` |
| **Reference** | `VoxtralTtsForConditionalGeneration.forward`'s own composition, with `x_0` pinned (see *Determinism*) |
| **Metric** | PCC(TT waveform, HF waveform) ≥ 0.99, plus **exact** equality of the emitted audio codes |

The chain (`tt/pipeline.py::run_tts`) — each stage is fed the previous **TT** stage's real output,
no reference tensor is injected at any joint:

```
ids + voice --[on-device embedding]--> inputs_embeds [1, P, 3072]
  -> tts_backbone                    -> hidden, last row h [1, 1, 3072]
  -> per frame: flow_matching(h)     -> 37 audio codes  (semantic argmax + 7 Euler steps + FSQ)
                stop if the semantic code is an [END_AUDIO] id
                embed_frame(codes)   -> [1, 1, 3072], appended to inputs_embeds
                tts_backbone         -> h               (HF's prefill_then_step, same arithmetic)
  -> codec_decoder(all frames)       -> waveform [1, 1, T*1920]
```

## Results (Blackhole p150b, single device, TP=1)

| Gate | Result |
|---|---|
| **Gate 1** — routed stubs still native | PASS — live probe: 0 torch ops in all 7 forwards |
| **Gate 2** — every graduated module invoked | PASS — `{tts_backbone: 9, r_m_s_norm: 18, attention: 9, m_l_p: 9, decoder_layer: 9, flow_matching: 8, codec_decoder: 1}` |
| **Gate 3** — e2e waveform PCC ≥ 0.99 | **PASS — `e2e PCC=0.9999834`** |
| audio-code flips vs reference | **0** of 296 codes |
| per-step hidden-state PCC | 1.000000 at every one of the 9 steps |
| `host_op_selftest` | fully on device — zero host aten ops in the model math |
| `trace_capture_selftest` | all 4 stages captured host-free and replayed: prefill C=224 pcc=1.000017, decode C=224 pcc=1.000000, flow C=3 pcc=1.000000, vocode C=32 pcc=0.999999 |
| trace replay of one decode frame | PASS — replay PCC 1.000000, **90.1 ms/frame** at C=224 over 26 layers |
| per-component PCC (7/7) | PASS (unchanged by the accuracy work below) |

Timings for the 8-frame gate: build ~30 s, prefill 2.2 s, decode 0.5 s/frame, codec 1.9 s.

Held at a longer horizon too — 24 frames (1.92 s of audio, 888 codes) via the demo:
`e2e PCC=0.9999762`, **0 code flips**.

## Run it

```bash
# demo — real WAV out (add --compare-reference to print the PCC too)
python -m models.demos.voxtral_tts_full.demo.demo_tts \
    --text "It took me quite a long time to develop a voice." \
    --voice neutral_male --max-frames 24 --out generated/voxtral_tts_tt.wav

# the e2e gate (Gates 1/2/3 + host-op + trace + layer-cap selftests)
./python_env/bin/python -m pytest models/demos/voxtral_tts_full/tests/e2e/test_tts_e2e.py -s

# trace-replay perf for one decode frame (prints TRACE_PER_TOKEN_MS)
./python_env/bin/python -m pytest models/demos/voxtral_tts_full/tests/e2e/test_tts_perf.py -s

# per-component PCC, as emitted by bring-up
./python_env/bin/python -m pytest models/demos/voxtral_tts_full/tests/pcc/ -s

# the two selftests as the tooling calls them: module-level, zero-arg, own device
./python_env/bin/python scripts/tt_hw_planner/_host_op_probe.py models/demos/voxtral_tts_full
./python_env/bin/python scripts/tt_hw_planner/_trace_capture_probe.py models/demos/voxtral_tts_full
```

`--text` omitted uses `config.default_prompt_ids` (the prompt the model ships, matched to the
`neutral_male` preset). The first reference run costs a few CPU-minutes and is then cached under
`_captured/e2e/`.

## Layout

```
demo/demo_tts.py          runnable entrypoint (argparse + __main__)   <- Call 1
demo/demo.py              bring-up scaffold (HF on CPU, ASR marker); superseded by demo_tts.py
tt/pipeline.py            THE chained forward pass + build_pipeline + stages + trace hooks
tt/reference.py           input encoding (tokenizer, voice preset) + the HF golden, cached
selftest_runtime.py       the ONE device open for the zero-arg selftests (outside tt/ on purpose)
tests/e2e/test_tts_e2e.py the gate
tests/e2e/test_tts_perf.py trace capture + replay of one decode frame (TRACE_PER_TOKEN_MS)
_stubs/*.py               the 7 graduated bodies (untouched)
tt_backbone.py            Block 1 building blocks shared by 4 of those stubs
tt_common.py              shared native-TTNN primitives
```

**The demo and the test share ONE pipeline.** Both import `tt/pipeline.py::build_pipeline` and
call `run_tts`; there is no second copy of the wiring, so a green test is a working demo by
construction.

## Where each graduated module runs

Bring-up graduated the same arithmetic at four granularities (`decoder_layer` was decomposed into
`r_m_s_norm` / `attention` / `m_l_p`). Calling only the coarsest would leave four graduated stubs
unused, and calling them on the side to tick a counter is a coverage sweep, not a forward path. So
the 26-layer stack is **composed out of them**, each built from its own layer's real weights:

| Backbone layer | Routed through | Built from |
|---|---|---|
| 0 | `r_m_s_norm` → `attention` → `r_m_s_norm` → `m_l_p` | `backbone.layers.0.{input_layernorm, self_attn, post_attention_layernorm, mlp}` — their own captured paths |
| 1 | `decoder_layer` | `backbone.layers.1` |
| 2–25 | `tts_backbone`'s own bodies | `backbone.layers.2..25` |

The list lives on the `tts_backbone` stub object, so calling that stub runs the whole mixed stack
plus its final norm. Nothing is computed twice. `flow_matching` runs once per frame and
`codec_decoder` once at the end.

## Decode horizon

Stop-token rule, applied identically to both sides: generation ends on an `[END_AUDIO]` semantic
code, accepting **both** ids the model uses (`voxtral_common_ref.END_AUDIO_ID = 1`, which the
graduated flow stub encodes, and `config.end_audio_id = 2048`, which HF's `forward` compares
against). The safety cap is `config.max_position_embeddings − prompt_len`.

The gate runs at **8 frames**, which is not a magic number: `_captured/codec_decoder/args.pt` is
`[8, 37]` and the captured KV width is 208 = 200 prompt + 8 frames, i.e. it is the horizon the
bring-up capture itself ran at. `captured_frame_count()` reads it from that file.

## Determinism (why the golden is spelled out instead of calling `forward`)

`voxtral_flow_ref.decode_frame` draws a fresh Gaussian `x_0` per frame when none is given, and
`forward` gives none. A probed forward cannot draw noise (`ttnn.from_torch` is 2 torch ops and
`native_probe` graduates at 0), so the graduated flow stub stages ONE `x_0` at build time — the
tensor the bring-up harness wrote to `_captured/flow_matching/x_0.pt`. `tt/reference.py` pins that
same tensor on the golden side; everything else is `forward` verbatim. Both sides then integrate
the same ODE, so any PCC gap is the port's arithmetic and nothing else.

## Accuracy engineering

Block 2's output is **quantised** (37 integer codes per frame), so error is not smoothly
forgiving: a dimension within ~1e-2 FSQ code units of a boundary flips a code, and a flipped code
changes the next frame's conditioning, which cascades. Measured on this board:

| | plain | split |
|---|---|---|
| matmul, any K (3 / 128 / 3072) | 1.1–1.8e-3 | **3.1e-4** (a three-term split measures the same — the floor is the accumulator, not the operands) |
| flow-block velocity | 7.9e-4 | **2.3e-4** |
| flow ODE final state | 1.13e-2 code units | **1.3e-3 code units** |
| 8-frame rollout | 67 flipped codes, waveform PCC 0.898 | **0 flips, 0.99998** |

So `tt_common.tt_matmul_hp` (both operands split hi/lo) is used for the two
activation×activation matmuls inside attention, and `tt_linear_hp` for every projection in the
backbone. Sensitivity for context: 10 random ±1 code flips already cost ~0.993–0.998 waveform PCC,
20 cost ~0.97.

Two properties this build relies on and asserts rather than assumes: the Mistral-native
interleaved RoPE is folded into the wq/wk rows at build time and checked against the real weights
(`verify_rope_permutation`), and the 37-way frame embedding is reduced with `mean * 37` (7e-8)
rather than `ttnn.sum` (1.1e-4).

The prefill is padded to a fixed capacity so the 26-layer stack sees one shape as the sequence
grows; causal attention makes that free (a real row can never see a padded one), and it is the
difference between 0.01 s and 1.75 s per frame in kernel compilation.

## Stages, trace and perf

`PIPELINE_STAGES = ["prefill", "decode", "flow", "vocode"]`, derived from the config:
`architectures = [...ForConditionalGeneration]` with an `AutoModelForCausalLM` mapping and no
`is_encoder_decoder` → `[prefill, decode]`; `modality_out = "audio"` → `+ [vocode]`; and
`block_stacks = ["backbone", "flow", "codec"]` declares three repeated stacks, so the flow block —
a distinct fixed-shape per-frame stage — gets its own entry and its own depth knob.

Each stage exposes `<stage>_trace_setup(inputs)` (pins the variable dim to a fixed capacity C and
pre-uploads the padded input; all shape-dependent constants — RoPE tables, causal mask, ALiBi and
window biases — are already build-time tables inside the stubs, taken from the model's own
`rope_theta` / `max_position_embeddings`), `<stage>_trace_step()` (one host-op-free forward at the
pinned shape) and a zero-arg `<stage>_trace_inputs()` that assembles that argument from the
captured reference tensors under `_captured/`. The AR stage also keeps the decode contract:
`decode_prefill(embeds)` seeds a resident KV cache (there is no cross-attention — this is a
decoder-only model) and `decode_step(emb)` reads it and never recomputes.

`build_pipeline(device, model=None, layers=None, **kwargs)` constructs and returns the resident
pipeline object; it never runs it. `layers` is the default depth for every repeated stack and
`<stage>_layers` overrides the stack that stage owns — `prefill_layers` / `decode_layers` (the
26-layer backbone), `flow_layers` (3), `vocode_layers` (the codec's per-stage transformer depth).
`None` means every layer; a cap never deletes a stack, and embeddings, norms and heads stay intact
so a capped build still exercises every distinct op. The repeated block is a plain list of
same-typed `TtBackboneStackLayer` elements (`pipe.backbone_layers`), and `pipe.hf` keeps the HF
reference reachable as ground truth for section structure.

Both selftests exist twice: as methods on the pipeline (what the pytest session calls, on the one
device the fixture opened) and as **module-level, zero-arg functions** in `tt/pipeline.py`, because
the observers import that module in a fresh process and call `host_op_selftest()` /
`trace_capture_selftest()` by name with nothing to pass them. In that case they get a device from
`selftest_runtime.py`, which lives *outside* `tt/` deliberately: the pipeline package must never
open a device of its own — a second ad-hoc open alongside the fixture's is a competing device with
a different command-queue count, which is what breaks trace. Both run at full depth.

`tests/e2e/test_tts_perf.py` is the measurement side of the same seam. Its unit is one decode
frame: `decode_trace_setup` seeds the resident KV cache and pins the step position, then exactly
`decode_trace_step()` — nothing else — is wrapped in `begin_trace_capture` / `end_trace_capture`,
checked against the eager result, and replayed. It reports `TRACE_PER_TOKEN_MS` and
`TRACE_REPLAY_PATH=trace+1cq`, which the perf harness records in
`test_tts_perf.py.trace_caps.json` — the artifact the trace gate reads.
