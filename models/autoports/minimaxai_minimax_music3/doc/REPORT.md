# MiniMax Music 3 on one Blackhole chip — bring-up report

generated 2026-09-10T02:12:49Z on qb2-120-p11t03

Model: `MiniMaxAI/MiniMax-Music3` @ `fbdf52fb` (8B Qwen3 global LLM + 0.6B depth decoder + 2.4B flow-matching DiT + DAC vocoder). Target: ONE Blackhole chip (P150; chip 0 of a P300/QB2). Package: tt-model-manager CONTAINER v5.1, kind `tt-dit-server`, profile `p150`.

## Verified

- CPU reference: the vendored torch pipeline reproduces the diffusers pipeline (text ids equal: True, exact code rows 1.000, audio PCC 1.0000)
- global LLM (1 chip, bfp8) teacher-forced on `readme` (201 steps, prompt 104 tokens): hidden PCC min 0.9962, logits PCC min 0.9971, c0 top-1 0.851 / top-5 1.000, top-1 vs sampled codes 0.418, 43.3 ms/step
- global LLM (1 chip, bfp8) teacher-forced on `blues` (301 steps, prompt 1380 tokens): hidden PCC min 0.9942, logits PCC min 0.9967, c0 top-1 0.811 / top-5 1.000, top-1 vs sampled codes 0.429, 39.0 ms/step
- long prompt prefill (3922 tokens): ok
- depth decoder (bf16) teacher-forced on `readme` (200 frames): logits PCC min 0.9984, c1..c7 top-1 0.942 / top-5 1.000, 53.6 ms/frame (7 traced steps + host)
- DiT (bf16): unit PCC min 0.9989; 30-step window: step-0 velocity PCC 0.9998, latents PCC 0.9996, vocoded audio PCC 0.999, spectral convergence 0.048; 147.1 ms per traced forward (L=689, batch 2)
- end-to-end `readme` on the chip: 200 frames, 8.0 s audio in 43.0 s (RTF 5.38); audio rms 0.098, silence 0.00, c0 unique ratio 0.80
- end-to-end `blues` on the chip: 300 frames, 12.0 s audio in 71.8 s (RTF 5.98); audio rms 0.078, silence 0.00, c0 unique ratio 0.56
- seeded generation is deterministic on the chip (identical codes for the same seed)
- bare-metal server (launcher env, p150): API proof passed (17/17 critical checks); speech RTF 4.509
- container `tt-model serve --profile p150` on this QB2: API proof passed (17/17 critical checks); speech RTF 3.718

## Not verified

- clean `tt-model pull` from the Hub + serve (p150): not run
- stage 11-report-push: not passed (see STATUS.md)
- multi-chip profiles (P300 / QB2 using >1 chip): not in scope; the package uses one chip everywhere
- listening-test quality of the songs: only proxy metrics (audio sanity, spectral distance vs the CPU reference) were measured; WAVs are in artifacts/e2e/ and artifacts/prove/

## Advisory findings and known limitations

- accuracy bars (from the CPU floors): {'c0_top1': 0.9, 'c0_top5': 0.98, 'depth_top1': 0.9, 'depth_top5': 0.98}; fp32 self-agreement of the SAMPLED goldens None (top-1 against sampled codes is < 1 even for the reference)
- `readme` c0 top-1 0.851 below the bar 0.9 (top-5 1.000); see the dtype sweep
- `blues` c0 top-1 0.811 below the bar 0.9 (top-5 1.000); see the dtype sweep
- dtype sweep selected {'llm_dtype': 'bfp8', 'depth_dtype': 'bfp8', 'depth_fidelity': 'hifi2', 'dit_dtype': 'bf16', 'dit_fidelity': 'hifi2'}; follow-ups: on-device c0/depth sampling (ttnn.sampling) to remove 8 host round trips per frame; single trace for the 7 depth steps; DiT window trace with on-device Euler/CFG/overlap; TT vocoder (conv1d); multi-chip: DiT on chip 1
- condition encoder and vocoder run on the CPU (torch fp32); scheduler/CFG/overlap math and sampling are on the host (phase A)
- the model's own contract limits: prompt <= 5 000 tokens, <= 9 000 frames (6 min); served max_seq_len 16 384

## Performance (single chip)

| component | variant | accuracy | speed |
|---|---|---|---|
| llm | bfp8 (selected) | c0_top1_min=0.811, c0_top5_min=1.000 | ms_per_step=52.0 |
| llm | bf16  | c0_top1_min=0.831, c0_top5_min=1.000 | ms_per_step=58.8 |
| depth | bf16  | depth_top1=0.940, depth_top5=1.000 | ms_per_frame=63.4 |
| depth | bfp8 (selected) | depth_top1=0.940, depth_top5=1.000 | ms_per_frame=49.6 |
| depth | bf16h2  | depth_top1=0.954, depth_top5=1.000 | ms_per_frame=63.2 |
| dit | bf16 (selected) | step0_pcc=1.000, latents_pcc=1.000, spectral_convergence=0.048 | ms_per_forward=157.1 |
| dit | bfp8  | step0_pcc=0.998, latents_pcc=0.998, spectral_convergence=0.080 | ms_per_forward=117.6 |
| dit | bf16h2  | step0_pcc=0.999, latents_pcc=0.999, spectral_convergence=0.047 | ms_per_forward=124.2 |

| clip | frames | prefill s | LLM ms/frame | depth ms/frame | denoise s | vocoder s | RTF |
|---|---|---|---|---|---|---|---|
| readme | 200 | 1.7 | 45.6 | 61.0 | 9.7 | 10.1 | 5.38 |
| blues | 300 | 0.7 | 46.5 | 60.5 | 18.6 | 20.1 | 5.98 |

## Stage table

| stage | body | gate | started | ended |
|---|---|---|---|---|
| 00e-metal-build | ok | 0 | 2026-09-09T23:49:40Z | 2026-09-09T23:58:41Z |
| 00-host-prep | ok | 0 | 2026-09-09T23:45:37Z | 2026-09-09T23:49:40Z |
| 01-cpu-reference | ok | 1 | 2026-09-10T00:56:31Z | 2026-09-10T01:35:05Z |
| 02-weights-config | ok | 0 | 2026-09-10T01:35:05Z | 2026-09-10T01:35:24Z |
| 03-llm-pcc-1chip | ok | 1 | 2026-09-10T01:40:29Z | 2026-09-10T01:41:10Z |
| 04-depth-decoder-tt | ok | 0 | 2026-09-10T01:43:48Z | 2026-09-10T01:44:08Z |
| 05-dit-tt | ok | 0 | 2026-09-10T01:44:08Z | 2026-09-10T01:44:52Z |
| 06-e2e-generator | ok | 1 | 2026-09-10T01:50:31Z | 2026-09-10T01:53:03Z |
| 07-optimize | ok | 1 | 2026-09-10T01:53:03Z | 2026-09-10T01:58:59Z |
| 08-server-smoke | ok | 0 | 2026-09-10T01:58:59Z | 2026-09-10T02:02:16Z |
| 09-package | ok | 0 | 2026-09-10T02:02:16Z | 2026-09-10T02:05:26Z |
| 10-serve-prove | ok | 0 | 2026-09-10T02:05:27Z | 2026-09-10T02:10:21Z |
| 11-report-push | running | - | 2026-09-10T02:12:45Z |  |

## Package

- manifest `/home/jashan/tt-model-builds/minimax-music3/tt_kernel_manifest.json` (schema 5.1, kind tt-dit-server, tt-metal 0.65.2.dev9785, image `tt-model/minimax-music3:6bf99951f162` sha256:6bf99951f162)
- weights pointer `MiniMaxAI/MiniMax-Music3` @ `fbdf52fbaaca` (ignore patterns skip the 28 GB of unconverted originals)
- HF push: huggingface_hub.errors.OfflineModeIsEnabled: Cannot reach https://huggingface.co/api/models/jashansinghTT/minimax-music3-blackhole: offline mode is enabled. To disable it, please unset the `HF_HUB_OFFLINE` environment variable.
- code: tt-metal worktree `~/tt-metal-music3`, branch `jashan/minimax-music3`, `models/autoports/minimaxai_minimax_music3`

## Artifacts

- goldens: artifacts/golden/ · PCC: artifacts/pcc/ · e2e audio: artifacts/e2e/ · perf: artifacts/perf/ · proofs: artifacts/prove/ · hardware events: logs/hw-events.log
