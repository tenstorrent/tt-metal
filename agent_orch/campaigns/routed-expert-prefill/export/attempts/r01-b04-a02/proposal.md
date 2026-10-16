# r01-b04-a02: halve the gate/up K-block count: in0_block_w_gu 16 -> 32, paid for in L1 by per_core_M_max 4 -> 2 (and revert the parent's read-ahead)

## Motivation
Every benchmark expert holds 68-272 tokens (3-9 tile-rows), so on the 8 M-rows the runtime picker
gives per_core_M = 1 (gpt-oss: 2) and always exactly ONE chunk. The CBs are sized for
per_core_M_max = 4, so 3/4 of the x staging (cb_x_rm 262 KB + cb_in0_x 70 KB on DS-V3) is reserved
and never used. Meanwhile the gate/up K-loop runs K/16 blocks (DS-V3/Kimi-K2.7/V4-pro 14,
GLM/M3 12, Mistral/V4-flash 8, K3 14 (w 8, see below), gpt-oss 6), and each block carries a fixed cost that
does not shrink with width:
- compute: tilize init + uninit + SrcA/pack reconfig + matmul_block_init + L1_ACC reconfig per
  block (RM path), and a packer STALLWAIT drain per block;
- dataflow: one receivers-ready round, one valid-sem multicast, one DRAM-read tail and one
  multicast tail per block for both x (row) and gate+up (column).
Evidence that the per-block cost is what's left: the two read-ahead nodes (r01-b03-a01,
r01-b04-a01) showed sender DRAM latency is not the limit, the measured ~33 cycles per tile-mult is
2x the LoFi rate (r01-b04-a01). The factory's own calibration note is mixed evidence: halving
the width via the M-doubling step (16 -> 8/4/2) cost 1.5-1.85x on most shapes at ISL 5120, but
on Kimi-K3 at ISL <= 256 (single-chunk experts) going from 7 to 14 blocks cost only ~3%. By my
replica of the fitter, Kimi-K3 currently runs (per_core_M_max 8, w 8) = 14 gate/up blocks
because it clears that step's bar; every other shape runs (4, 16).

## Mechanism
Host only (`unified_routed_expert_ffn_program_factory.cpp`):
- requested per_core_M_max 4 -> 2 (kMaxChunkMTiles = 2 * GRID_Y), requested in0_block_w_gu
  16 -> 32. The x staging CBs keep the same tile count (M_max * w = 64), the gate/up weight CBs
  double. The existing L1 fitter then picks the widest divisor of K_gate that fits:
  DS-V3/Kimi-K2.7/GLM/Mistral/V4-flash 32, K3 28, V4-pro 28, M3 24, gpt-oss 30
  (from a Python replica of cb_footprint_bytes; all fit with >100 KB to spare).
- drop the "double per_core_M when it saves enough weight traffic" step. It only pays when an
  expert has more tile-rows than one chunk; with the runtime picker that never happens at the
  production per-chip loads, and it would undo the widening (it picks (4,16) back on every shape).
- revert the parent's (r01-b04-a01) read-ahead in reader/writer to root code, since it measured
  a regression.
No kernel change: all kernels take in0_block_w_gu / per_core_M_max as compile-time args.

## Why this is not a repeat
r01-b03-a01 / r01-b04-a01 changed when the sender issues DRAM reads (both regressed). r01-b01-a01 /
r01-b02-a01 changed which rows run which expert. Nobody changed the K-block geometry. This
attempt cuts the number of K-block handshakes/reconfigs instead of overlapping them, and it is
orthogonal to (and combinable with) the row-group expert parallelism.

## Expected effect and risk
If per-block fixed cost is ~half of gate/up time, gate/up time drops by ~25% and the op by
~10-20% (gate/up is ~2/3 of the MACs; the down phase is untouched). Biggest on the 14-block
shapes. Risks: bigger first block per expert means a longer pipeline fill per expert; for
experts with more than 16 tile-rows (512 tokens) the halved chunk size doubles weight re-reads
(not exercised here; a runtime two-mode width is the general fix). Accuracy: no format, fidelity or
dst-accumulation mode changes, but the packer L1_ACC accumulation into the bf16 partials now
happens in 7 instead of 14 steps (more of the K sum stays in dst), so the rounding of the partial
sums changes slightly; the relative accuracy gate will show whether
that matters (it should be neutral or slightly better). L1/CB overflow would show as a host
TT_FATAL or a hang.
