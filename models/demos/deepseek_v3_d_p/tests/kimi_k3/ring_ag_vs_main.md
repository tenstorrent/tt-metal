# Ring gather / ring joint SDPA: this branch vs main

Measured 2026-10-16 on bh-lb-09 (8x p150b LoudBox, mesh 2x4, SP4 x TP2, 2 links, Linear), best of 3, with
`test_zz_ring_probe.py`. Driver: `ring_ag_vs_main.sh -m <main worktree>`. The machine ran the Galaxy-like firmware
(12x10 Tensix grid, 200G eth) at the time; standalone gathers there are ~1.5x slower than on stock firmware
(fabric_all_gather too), so compare rows, not absolute numbers with the 2026-10-08/09 runs.

Trees: main = origin/main 98d48c1d276; stock base = ee673c4ded8 (this branch's parent); branch =
mstaletovic/ring-ag-fab-overlap with `RING_AG_PREFETCH_PACKETS=8`. Layout C = 11x9 attention on rows 1-9 with the
gather workers on the eth-nearest top-row cores (`RING_PROBE_PLACEMENTS=11x9@0,0r RING_SDPA_GRID_ORIGIN_Y=1
RING_AG_PLACE_ETH_FUSED=1`); stock code rejects that grid ("SDPA coregrid overlaps with AllGather coregrid").

Expanded outputs are bit-identical in every tree (checksum sum -78.514038 abs 16089.784180, gathers max|err| 0).

## Expanded K/V (24 heads/chip, q32 / k640, K192 + V128, 57.4 MB K+V shard per chip): ring-bound

| Tree | Fused ring_joint_sdpa | Expanded gather alone |
|---|---|---|
| stock base, 10x10 + CCL column | 7303 us | 4535 us |
| **main, 10x10 + CCL column** | **7376 us** | 4833 us |
| branch, 10x10 + CCL column | 6499 us (-12% vs main) | 4086 us (-15%) |
| **branch, layout C** | **4861 us (-34% vs main)** | 4087 us |
| fabric_all_gather (reference) | - | 3722 us |

## K3 warm latent (ring_mla, 24 heads x 640 q rows per chip, 5K chunk): attention-bound

| Tree | Fused ring_mla | Latent ring gather alone |
|---|---|---|
| stock base | 4777 us | 374 us |
| main | 4784 us | 378 us |
| branch, 10x10 | 4780 us | 343 us (-9%) |
| branch, layout C | 4759 us | 343 us |

At the production K3 shape the gather is hidden under ~4.7 ms of attention, so the branch gains nothing there; it pays
off where the ring is on the critical path (expanded K/V, larger KV rows, small chunks).
