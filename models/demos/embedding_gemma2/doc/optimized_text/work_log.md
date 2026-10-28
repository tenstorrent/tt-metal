# E5a work log (optimized text tower)

Host sjc-snva-tp100, one P150a, branch `gtobarTT/embedding_gemma2_bringup`, base E4 `9e1cb0dd79`.
Env: `source tt-ops/runs/embeddinggemma-2/e1/env.sh` plus, from 15:47 on,
`HF_MODEL=<HF cache>/snapshots/914f7f89142e33e77833254d9c9b90c3cef7303b HF_HUB_OFFLINE=1`
(`e5a/scripts/pin.sh`; see "Environment incident"). Logs, sweeps and raw data:
`tt-ops/runs/embeddinggemma-2/e5a/`. Times are local (PDT), 2026-10-27.

## Timeline

| time | step | result |
|---|---|---|
| 14:59 | baseline re-measure at E4 (`bench_sustained.py`, label `baseline_E4`) | 1x128 8.752, 1x512 18.314, 1x2048 58.984, 8x512 74.498 ms burst; sustained equal within 0.015 ms |
| 15:05-15:30 | SDPA sweeps (`sdpa_sweep.py`, `sdpa_variants.py`) at 1x128 / 1x512 / 1x2048 / 8x512 | 2.4-4.4x per SDPA call from chunk/grid choice + HiFi2; native window and windowed mode slower than the dense mask |
| 15:24 | c1 SDPA table + HiFi2 + `output_concat_heads`; text tests 98 passed | commit `ebe6ede2c1` |
| 15:20-15:35 | matmul sweep (`mm_sweep.py`, 4 M values x 10 roles x ~60 candidates) | 2D multicast in0_block_w 8 halves the default; bfp4 PCC 0.993 -> rejected |
| 15:40 | c2 projections: first run 11 failed at M >= 4096 (static CBs beyond L1) | added a CB budget (900 KB) with fallback to minimal_matmul; rerun 98 passed |
| 15:42 | precision A/B (fp32 acc vs not, bf16 vs bfp8 weights) on `test_text_model` | kept bfp8 + HiFi2 + fp32 acc (cosine 0.99751); bf16 weights 0.99938 but 2-7.5 % slower |
| 15:47 | environment incident (HF hub revision moved), pinned the snapshot | see below |
| 15:50 | c4 L1 intermediates: CB clash at S = 8192 | added `L1_ACT_MAX_TOKENS = 4096`; 98 passed |
| 15:58 | c3 (L1 off) regressed 1x2048 / 8x512; Tracy c4 showed minimal_matmul with fused gelu at 218 us | routed activation matmuls to the 2D config |
| 16:10 | c5 A/B: gelu placement x residual in L1 (burst) | residual L1 kept; gelu in the multiply only <= 1024 tokens |
| 16:12 | commit `93c41d6bce` (c2-c5) | 1x512 7.67 ms |
| 16:20 | RoPE micro in a trace (`rope_micro.py`) | hf / llama variants no faster -> rejected; L1 tables kept |
| 16:25-16:35 | block map at 4096 / 8192 | 2.2x / 5x per sliding SDPA |
| 16:50 | c6 bfp4 masks + block map, c7 L1 RoPE tables, 98 passed; commit `29efe95c78` | 1x8192 175.5 -> 109.7 ms |
| 17:00 | 65536 probe (`long_ctx.py`); commit `93f4074a48` | runs; padding equivalence cosine 1.000000 |
| 17:05-17:25 | gates on `93f4074a48` | all pass (demo after restoring the HF ref, see below) |
| 17:30 | Tracy before/after, final bench | tables in README |

## Environment incident: HF hub revision

At 15:47 a smoke test resolved `google/embeddinggemma-2` online. The hub had a new revision
`21ad7b3a49`, so `resolve_snapshot` downloaded it with its allow-patterns (`*.json`,
`*.safetensors`, `tokenizer*`). That snapshot has no `README.md` (the test corpus) and no
`chat_template.jinja` (the multimodal demo), so every test failed with FileNotFoundError and the
multimodal demo with "processor does not have a chat template". The weights blob and
`config.json` are identical (same blob hashes) **[M]** `ls -la` of both snapshot dirs.
Action: all E5a runs pin the E1-E4 snapshot (`HF_MODEL=<path>`, `HF_HUB_OFFLINE=1`), and before
the demo gate `refs/main` in the HF cache was set back to `914f7f89...` (previous value saved in
`e5a/logs/hf_refs_main_before_restore.txt`). Without `HF_HUB_OFFLINE=1`, the next online resolve
moves it again. Fix candidates for the orchestrator: pin the revision in `resolve_snapshot` or the
env (`HF_HUB_OFFLINE=1`), or add `*.md` / `*.jinja` to its allow-patterns.

## Notes

- The eager micro-benchmarks are host bound for ops under about 20 us (Python dispatch); the
  RoPE micro and every kept/rejected end-to-end decision use traced timing or Tracy device time.
- Tracy is not run with watcher; the watcher run is separate (gate G4).
- `capture_all` contract unchanged: masks, block maps and L1 RoPE tables are uploaded in
  `TextModel.prepare(S)`, before any capture.
