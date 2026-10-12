# t454: fix FastH3 Turbo SLA attention mode (run 1713, 2026-10-12 PDT)

Branch ttp/t454-fix-fasth3-turbo-sla-attention-mode-agai, pushes onto ttp/t452-continue-417-turbo-sla-vs-vsa-a-b-on-exa.

## Done
- LightX2V SLA inference (lightx2v-src/: sla_util.py, dsa.py, ti.py): dynamic_sparse_attn + sage2 is pure block-sparse
  attention (get_block_map: smooth-K, mean pool, unscaled scores, floor top-k; BLKQ 128 / BLKK 64 off sm90). There is
  NO linear-attention branch at inference, and MiniMax-H3's LightX2V attention has no gate_compress either (ours is
  off by default: gate_compress=False, so no o_c). Text/aux tokens are part of the sparse sequence when SP is off
  (sp_state None); with LightX2V SP the aux tokens go through calculate_parallel separately (not checked yet).
- Commit 48614457ac0: models/tt_dit/tests/models/minimax_h3/test_sla_attention_minimax_h3.py. Replays our stage's
  device-path math per SP shard (the stage's own a_t_q/a_t_kv with padded pooling, gathered K, cand_mask, top-k k_pad,
  kernel padded->global id map, block_counts) plus block-sparse attention, vs LightX2V. Result: 0 selection mismatches
  and output equal to 1e-16 at the real job geometry (75449 rows, SP=8 (job 024 log: TP=4/SP=8), q_block_tiles 2:
  1184 tiles, 148/shard, 256 slots, k=176). So the HOST math (geometry, stage, id mapping) is NOT the bug.
- Run it on g15blx02 (no matching _ttnn build there; the stub supplies names only):
  `PYTHONPATH=tt-project/t454/stub:. /home/smarton/fasth3/tt-metal/python_env/bin/python tt-project/t454/runtests.py`

## Remaining suspects (device side)
1. vsa_sdpa streaming kernel with n_exempt=0, all-zero dense_row_mask, list_len=176 (k_pad=176, W=176), SP=8
   two-op path: no device unit test covers SLA-shaped args. tests/ttnn/unit_tests/operations/sdpa/test_vsa_sdpa.py
   test_vsa_sdpa_raw_selection_matches_assembled can be parametrized with exempt_ids=[] and padded ids.
2. topk_large_indices on [1,H,256,2048] bf16 rows with many -inf (k=176): check ordering/ids on device.
3. bf16 selection (scores in bf16) is unlikely to cause 12 dB noise.
4. Pipeline: any VSA-structure assumption (n_video_tiles=0, n_prefix_tiles=all) in rope/AdaLN/output via
   permute_metadata/untile_index; dense rung 76800 vs geometry.padded_len 75776.
Best next step: one blx01 device job running a single SLA attention layer (random or real Q/K/V, SP=8 full mesh)
through MiniMaxH3VSACoarseStage + vsa_sdpa and comparing with _lightx2v_sla from the new test (dump indices too).
That isolates kernel vs pipeline. Job 024 overlay: /var/tmp/fasth3/t452/w (53f7b6b707 + sla.patch 7a224680d5ee).
