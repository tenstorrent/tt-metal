# r02-b03-a04: port r02-b01-a03's pack-thread activation pipeline onto the best lineage: the op-local fast SFPU bodies run on PACK, overlapped with the last gate/up K-block's matmuls

## Motivation
Two independent compute-side wins exist on different branches and have never been combined:
- r02-b01-a03 (+2.6% over its parent): moves the gate/up activation's SFPU work to the PACK
  thread and pipelines it into the last gate/up K-block, so subblock s is activated while MATH
  computes subblock s+1's last-block matmuls. Measured per shape: minimax-m3 -5.2%, gpt-oss -3.2%,
  glm -2.3%, kimi-k2.7 -1.5%, kimi-k3 -1.0%, dsv4-pro -0.9%.
- r02-b03-a03 (this lineage, best 1.5435): same-math op-local SFPU bodies that issue ~30% fewer
  instructions, but still run serially on MATH after the K-loop (`binary_activation_phase`), and
  the SiLU shapes still run silu_tile serially in the gate copy pass.
The two touch different things: a03 shortens the SFPU body, b01-a03 hides it behind FPU work.
The hidden part is bounded by the last block's FPU time per subblock, so a shorter body does not
remove the overlap win (it just finishes inside it more often). This lineage also has the
mixed-height layouts, where gpt-oss runs one expert per 3/2-row group at pcM 3-4, i.e. more
last-block subblocks to overlap with (r02-b01-a03 next #1 predicted ~1.55-1.56 for the combo).

## Mechanism
Compute kernel only (`device/kernels/compute/fused_swiglu.cpp`; comment touch-up in
`binary_act_sfpu.h`). No host, reader or writer change, so the schedule (two-phase + mixed
layouts) and every CB size are unchanged.
1. 3-way apply of r02-b01-a03's `matmul_phase_fused_gu` diff (that function is byte-identical
   between r02-b01-a02 and this lineage): the last K-block is walked in LW-wide subblocks; each
   subblock's partials are pushed after its last-block pack; before the next subblock's matmul,
   MATH copies (or, gpt-oss, bias-adds) the previous subblock's partials into dst, and PACK
   waits on the math commit, points the SFPU at its dst half, runs the activation, waits for
   SFPU idle and packs to cb_activated (binary ops) or cb_gate_intermed (SiLU). Pads keep every
   CB's block size constant as before. `binary_activation_phase` is deleted.
2. The binary op called on PACK is this lineage's op-local `re_act::binary_act` (same template
   args as on MATH today), via `SFPU_BINARY_CALL` wrapped in PACK(); its init
   (`re_act::binary_act_init`: stock table init + Prgm1/2 constants) moves to PACK at kernel
   start. `binary_act_sfpu.h` is now included on TRISC_PACK only. SiLU uses silu_tile_pack /
   silu_tile_init_pack (same calculate_silu as silu_tile).
3. Inert MATH-thread zones RE-GU / RE-ACT / RE-DN (as in r02-b01-a02 / r02-b01-a03) for the
   reflection's phase split.

Replay buffer: each TRISC has its own replay buffer. On PACK, only PackMode::Tilize records into
it, and the tilize helper (InitAndUninit) re-records it on every owner tilize, which never runs
inside the last block. sfpi records its replayed body on each call, so nothing depends on stale
slots. MATH's matmul replay region is no longer overwritten by SFPU code at all.

## Why this is not a repeat
- r02-b01-a03 did this on the r02-b01 lineage (uniform layouts, stock SFPU bodies). This node is
  the combination with r02-b03-a03's bodies and r02-b03-a02's schedule; it was listed as the
  top next step by r02-b01-a03 and is untried.
- r02-b02-a03 / r02-b04-a02 moved DM around the activation gap (both lost); this changes only
  compute threads.

## Expected effect and risk
- Expected: gpt-oss and minimax-m3 -2..-4% (their activation is already ~30% shorter, so a bit
  less than b01-a03's -3/-5%), glm / kimi-k2.7 / mistral -1..-2% (SiLU overlap), DRAM-bound
  shapes (dsv3, kimi-k3, dsv4-pro) ~0..-1%. Geomean about +1..+2% -> ~1.56-1.57.
- Numerics must stay bit-identical to the parent (same SFPU bodies, same pack/L1_ACC order).
  Any PCC/mean_abs change means a hand-off or format bug.
- Risks: PACK-thread replay interplay (would show as garbage/hang), CB pad mismatch (hang), PACK
  init clobbering pack state (accuracy). The local JIT harness compiles all four activation
  variants on all three TRISCs, row-major and TILE x.
