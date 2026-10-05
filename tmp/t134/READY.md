# #134 blx03 A/B: LTX_SDPA_MM_LOFI (ready, NOT launched)

Branch `ttp/t134-cherry-pick-sdpa-chain-57979-58032-58223`: t48 tip c4409b1fa24 + #57979, #58032, #58223
cherry-picks + the opt-in knob `LTX_SDPA_MM_LOFI=1` (ring SDPA QK^T and softmax @ V matmuls at LoFi; the
rest of the SDPA kernel stays HiFi2). Knob unset passes no kwarg, so the configs match the base exactly.

## Launch (only after the user lifts the device STOP-ALL; one project job at a time)

1. blx03 build: `ssh g14blx03 'grep SETUP134_DONE ~/fasth3/t134-setup.log'` must show `rc=0`, built from
   the branch's current tip. If missing, failed or older than the tip, copy `setup134.sh` to
   `~/fasth3/t134-setup.sh` on blx03 and run it detached
   (`setsid nohup bash ~/fasth3/t134-setup.sh > ~/fasth3/t134-setup.log 2>&1 < /dev/null &`); it fetches
   the tip, checks it out in ~/fasth3/t134 and builds incrementally.
2. `tt-project/harness/templates/blx03-launch.sh t134 /home/smarton/fasth3/t134/tmp/t134/driver.sh`
   and hand off waiting with the retry_when it prints (`survives_reboot: true`).

The driver queues two short broker jobs through blx03's tt-device-mcp (submit.sh: one project job at a
time, bare-2x4 scan), each opening the full (4,8) mesh and then `create_submesh(ttnn.MeshShape(2, 4))`:

| job | build | arms | broker timeout |
|-----|-------|------|----------------|
| ref | ~/fasth3/t48 (base, no cherry-picks, no knob) | REF | 1200 s |
| new | ~/fasth3/t134 (this branch) | OFF, LOFI | 1800 s |

Expected run time is ~3 min (ref) and ~6-10 min (new, cold kernel cache), going by #81's 3 min job.

## Output (blx03 /var/tmp/fasth3/t134/)

- `driver.log`: job ids, health gates, `T134_REF_DRIFT` if ~/fasth3/t48 moved off the base, and
  `T134_DRIVER_DONE <stage> <rc>`. Stage `*_drop` (rc 9) means a drop, ERROR or reboot during our job:
  stop ALL device work on every galaxy and report it.
- `run134_ref.log`, `run134_new.log`: full pytest logs. `summary.txt`: the `T134_*` lines.

Lines, at S2 (19,34,60) on the Linear 2x4 sp1/tp0 layout, AV block 0 of the 22B checkpoint:
- `T134_BLOCK arm=X S2 ms_per_block=` traced block time (median of 3 laps x 10 replays).
  #81's 2x4 S2 block was 60.295 ms.
- `T134_RING ... ms_per_call=` each ring SDPA call of the block (video self, V2A cross) traced alone;
  `T134_RING_SUM ... ring_ms= share=` their sum. The 4x8 baseline is 5.37 of 16.4 ms per S2 block;
  2x4 carries ~4x the ring SDPA work per chip.
- `T134_CMP S2 OFF_vs_REF` / `LOFI_vs_REF` / `LOFI_vs_OFF`: identical flag, PCC, PSNR
  (20*log10(max|ref|/rmse)), max abs, for video and audio block outputs. `T134_RING_CMP`: isolated ring
  SDPA output, LOFI vs OFF. `T134_AB S2`: block and ring deltas, LOFI vs OFF.
- `T134_TORCH arm=X`: video-only block vs the diffusers torch block at S1 (PCC, rel RMSE);
  `T134_TORCH_CMP`: the same outputs vs REF and LOFI vs OFF.
- `T134_GATE off_identical_to_ref_S2=True` is the default-path check (knob off must be bit-identical).

## Reading it

- OFF vs REF not identical (with no `T134_REF_DRIFT`): a cherry-pick moved the default path. The
  default-path kernel helpers (sdpa_mm_* in compute_streaming.hpp) reduce to the old calls without the
  define, and the new merge helpers only run with K split or segmented accumulation, so this is not expected.
- OFF vs REF ring/block ms: what the cherry-picks alone (multicast halo, batched head-op NoC) give.
- LOFI: keep only if `T134_AB` shows a real ring SDPA cut and the block PSNR/PCC hold; the LoFi
  numerics then need the e2e quality pass (PSNR first, VBench, 5 seeds when in doubt) before it can
  become a default.
- Knobs: `T134_SHAPES=S1,S2`, `T134_REPLAYS`, `T134_RING=0` (skip isolated ring timing),
  `T134_TORCH=0` (skip phase B).
