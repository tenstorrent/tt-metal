# MiniMax Music 3 on one Blackhole chip — bring-up report

generated 2026-09-10T02:56:26Z on qb2-120-p11t03

Model: `MiniMaxAI/MiniMax-Music3` @ `fbdf52fb` (8B Qwen3 global LLM + 0.6B depth decoder + 2.4B flow-matching DiT + DAC vocoder). Target: ONE Blackhole chip (P150; chip 0 of a P300/QB2). Package: tt-model-manager CONTAINER v5.1, kind `tt-dit-server`, profile `p150`.

## Verified

- CPU reference: the vendored torch pipeline reproduces the diffusers pipeline (text ids equal: True, exact code rows 1.000, audio PCC 1.0000)
- global LLM (1 chip, bfp8) teacher-forced on `readme` (201 steps, prompt 104 tokens): hidden PCC min 0.9962, logits PCC min 0.9971, c0 top-1 0.851 / top-5 1.000, top-1 vs sampled codes 0.418, 43.3 ms/step
- global LLM (1 chip, bfp8) teacher-forced on `blues` (301 steps, prompt 1380 tokens): hidden PCC min 0.9942, logits PCC min 0.9967, c0 top-1 0.811 / top-5 1.000, top-1 vs sampled codes 0.429, 39.0 ms/step
- long prompt prefill (3922 tokens): ok
- depth decoder (bf16) teacher-forced on `readme` (200 frames): logits PCC min 0.9984, c1..c7 top-1 0.942 / top-5 1.000, 53.6 ms/frame (7 traced steps + host)
- DiT (stage-05 run, bf16/HiFi4; the served default is bf16/HiFi2 at 93 ms, see the performance table): unit PCC min 0.9989; 30-step window: step-0 velocity PCC 0.9998, latents PCC 0.9996, vocoded audio PCC 0.999, spectral convergence 0.050; 137.2 ms per traced forward (L=689, batch 2)
- end-to-end `readme` on the chip: 200 frames, 8.0 s audio in 32.8 s (RTF 4.10); audio rms 0.053, silence 0.00, c0 unique ratio 0.78
- end-to-end `blues` on the chip: 300 frames, 12.0 s audio in 48.5 s (RTF 4.04); audio rms 0.091, silence 0.00, c0 unique ratio 0.64
- seeded generation is deterministic on the chip (identical codes for the same seed)
- bare-metal server (launcher env, p150): API proof passed (17/17 critical checks); speech RTF 2.983
- container `tt-model serve --profile p150` on this QB2: API proof passed (17/17 critical checks); speech RTF 2.988
- clean `tt-model pull` from the Hub + serve (p150): API proof passed (11/11 critical checks); speech RTF 2.949

## Not verified

- multi-chip profiles (P300 / QB2 using >1 chip): not in scope; the package uses one chip everywhere
- listening-test quality of the songs: only proxy metrics (audio sanity, spectral distance vs the CPU reference) were measured; WAVs are in artifacts/e2e/ and artifacts/prove/

## Advisory findings and known limitations

- accuracy bars (from the CPU floors): {'c0_top1': 0.9, 'c0_top5': 0.98, 'depth_top1': 0.9, 'depth_top5': 0.98}; fp32 argmax vs the SAMPLED golden codes {'c0_argmax_top1': 0.3880597014925373, 'c0_top5': 0.7661691542288557, 'depth_argmax_top1': 0.22103766879886283, 'depth_top5': 0.4896943852167733}; bf16-CPU vs fp32 {'c0_top1': 0.9552238805970149, 'c0_top5': 1.0, 'depth_top1': 0.9566453447050463, 'depth_top5': 1.0, 'hidden_pcc_min': 0.9993515014648438, 'c0_logits_pcc_min': 0.9997697472572327} (top-1 against sampled codes is < 1 even for the reference)
- `readme` c0 top-1 0.851 below the bar 0.9 (top-5 1.000); see the dtype sweep
- `blues` c0 top-1 0.811 below the bar 0.9 (top-5 1.000); see the dtype sweep
- dtype sweep selected {'llm_dtype': 'bfp8', 'depth_dtype': 'bfp8', 'depth_fidelity': 'hifi2', 'dit_dtype': 'bf16', 'dit_fidelity': 'hifi2'}; follow-ups: on-device c0/depth sampling (ttnn.sampling) to remove 8 host round trips per frame; single trace for the 7 depth steps; DiT window trace with on-device Euler/CFG/overlap; TT vocoder (conv1d); multi-chip: DiT on chip 1
- condition encoder (fp32) and vocoder (bf16, PCC 0.99994 vs fp32) run on the CPU; scheduler/CFG/overlap math and sampling are on the host (phase A)
- the from-the-Hub re-serve pulled the manifest/image from the repo, but docker found the identical image blobs already loaded on this box (same digest), so the image download itself was not exercised here
- the model's own contract limits: prompt <= 5 000 tokens, <= 9 000 frames (6 min); served max_seq_len 16 384

## Performance (single chip)

| component | variant | accuracy | speed |
|---|---|---|---|
| llm | bfp8 (selected) | c0_top1_min=0.811, c0_top5_min=1.000 | ms_per_step=52.0 |
| llm | bf16  | c0_top1_min=0.831, c0_top5_min=1.000 | ms_per_step=58.8 |
| depth | bf16  | depth_top1=0.944, depth_top5=1.000 | ms_per_frame=47.8 |
| depth | bfp8 (selected) | depth_top1=0.949, depth_top5=1.000 | ms_per_frame=37.3 |
| depth | bf16h2  | depth_top1=0.948, depth_top5=1.000 | ms_per_frame=46.0 |
| dit | bf16 (selected) | step0_pcc=1.000, latents_pcc=1.000, spectral_convergence=0.050 | ms_per_forward=138.8 |
| dit | bfp8  | step0_pcc=0.998, latents_pcc=0.998, spectral_convergence=0.076 | ms_per_forward=92.8 |
| dit | bf16h2  | step0_pcc=0.999, latents_pcc=0.999, spectral_convergence=0.047 | ms_per_forward=93.2 |

| clip | frames | prefill s | LLM ms/frame | depth ms/frame | denoise s | vocoder s | RTF |
|---|---|---|---|---|---|---|---|
| readme | 200 | 1.5 | 36.4 | 62.1 | 8.3 | 3.3 | 4.10 |
| blues | 300 | 0.7 | 37.3 | 46.4 | 16.2 | 6.4 | 4.04 |

## Stage table

| stage | body | gate | started | ended |
|---|---|---|---|---|
| 00e-metal-build | ok | 0 | 2026-09-09T23:49:40Z | 2026-09-09T23:58:41Z |
| 00-host-prep | ok | 0 | 2026-09-09T23:45:37Z | 2026-09-09T23:49:40Z |
| 01-cpu-reference | ok | 1 | 2026-09-10T00:56:31Z | 2026-09-10T01:35:05Z |
| 02-weights-config | ok | 0 | 2026-09-10T01:35:05Z | 2026-09-10T01:35:24Z |
| 03-llm-pcc-1chip | ok | 1 | 2026-09-10T01:40:29Z | 2026-09-10T01:41:10Z |
| 04-depth-decoder-tt | ok | 0 | 2026-09-10T01:43:48Z | 2026-09-10T01:44:08Z |
| 05-dit-tt | ok | 0 | 2026-09-10T02:21:21Z | 2026-09-10T02:22:29Z |
| 06-e2e-generator | ok | 1 | 2026-09-10T02:41:40Z | 2026-09-10T02:43:33Z |
| 07-optimize | ok | 1 | 2026-09-10T02:43:33Z | 2026-09-10T02:46:45Z |
| 08-server-smoke | ok | 0 | 2026-09-10T02:46:45Z | 2026-09-10T02:49:02Z |
| 09-package | ok | 0 | 2026-09-10T02:49:02Z | 2026-09-10T02:52:14Z |
| 10-serve-prove | ok | 0 | 2026-09-10T02:52:14Z | 2026-09-10T02:54:31Z |
| 11-report-push | running | - | 2026-09-10T02:54:31Z |  |

## Package

- manifest `/home/jashan/tt-model-builds/minimax-music3/tt_kernel_manifest.json` (schema 5.1, kind tt-dit-server, tt-metal 0.65.2.dev9789+g5039f8a95a6, image `tt-model/minimax-music3:9abe0907aae5` sha256:9abe0907aae5)
- weights pointer `MiniMaxAI/MiniMax-Music3` @ `fbdf52fbaaca` (ignore patterns skip the 28 GB of unconverted originals)
- HF push:   → consumers:  tt-model serve jashansinghTT/minimax-music3-blackhole
- code: tt-metal worktree `~/tt-metal-music3`, branch `jashan/minimax-music3`, `models/autoports/minimaxai_minimax_music3`

## Artifacts

- goldens: artifacts/golden/ · PCC: artifacts/pcc/ · e2e audio: artifacts/e2e/ · perf: artifacts/perf/ · proofs: artifacts/prove/ · hardware events: logs/hw-events.log
