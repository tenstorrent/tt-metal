# Padded BF16 width-4 pack qualification

This is an isolated hypothesis test, **not yet device-qualified**. No attention
header, existing probe, frozen control, chunk size, or input-buffer count changed.

## Audit conclusion

The standard default width-4 MOP cannot honor a 4096-byte BF16 tile pitch by
changing a stride register alone. It closes the output stream only once, after
all sixteen faces. The output address generator computes new addresses only
after PACR Last or Flush; otherwise the BF16 byte stream appends contiguously.
The CB page size is used for the initial scalar output address, not automatically
at the four internal tile boundaries.

A custom MOP can express the required layout using documented mechanisms:

- Leave DST/input (channel 0) strides and ADDR_MOD_0/1/2 as standard Default pack.
- Set output channel 1 Y stride to 64 bytes; X stride remains zero.
- Replay the first 15 PACRs of one tile, with ADDR_MOD_2 at face boundaries and
  ADDR_MOD_0 elsewhere. They advance output Y from 0 to 60.
- The sixteenth PACR closes this tile with Last=1 and ADDR_MOD_2: output Y becomes
  64, source Y becomes 0, source Z advances to the next tile.
- Run that tile pattern four times in one MOP. The final PACR of the fourth tile
  instead uses ADDR_MOD_1, restoring Ysrc/Ydst/Zsrc as standard pack does.
- The four tile-start addresses are base+[0,64,128,192]*64 bytes, i.e.
  base+[0,4096,8192,12288]. Each tile emits only its 2048-byte BF16 payload.
- Restore channel-1 Y stride to zero for ordinary compact output. Default
  pack_init does not restore channel-1 strides, so omitting this is unsafe.

This uses one C++ pack_tile call and one base-address setup per four tiles,
but **still four tile-close/drain events**. It therefore cannot promise the
same performance as contiguous width-4 packing. The custom MOP adds replay
dispatches; format/MOP/stride transitions can also dominate.

The private probe loads the 15-instruction pack replay once per invocation.
Standard Default pack initialization used by its other stage does not touch
that replay. Attention integration would require separately auditing replay
ownership, restoring stride during every format transition, and block-width
cache invalidation. This probe does not establish those properties.

## Sources

Local source paths, also pinned into every JSON result:

- tt_metal/tt-llk/tt_llk_blackhole/llk_lib/llk_pack.h:
  default address modifiers/MOP, pack_init, and pack execution.
- tt_metal/tt-llk/tt_llk_blackhole/common/inc/cpack_common.h:
  set_packer_strides programs channel 0 for Default; program_packer_destination.
- tt_metal/tt-llk/tt_llk_blackhole/common/inc/ckernel_template.h:
  replay plus terminal inner/outer instructions.
- tt_metal/hw/ckernels/blackhole/metal/llk_api/llk_pack_common_api.h:
  scalar CB page-stride address calculation.

The public ISA describes the output stream and explicitly distinguishes
Blackhole byte-address stride from Wormhole units:
https://github.com/tenstorrent/tt-isa-documentation/blob/main/WormholeB0/TensixTile/TensixCoprocessor/Packers/OutputAddressGenerator.md

These are architectural building blocks, not an existing supported high-level
blocked-padded pack API. Actual Blackhole correctness/performance remains to test.

## Probe and commands

Run from the worktree root with the device Python environment:

```sh
python experiments/sdpa-l2/bfp4-lofi-v2/padded_pack_probe.py --label padded-pack-bf16-smoke --tiles 128
python experiments/sdpa-l2/bfp4-lofi-v2/padded_pack_probe.py --label padded-pack-fp32-smoke --tiles 128 --fp32-dst
python experiments/sdpa-l2/bfp4-lofi-v2/padded_pack_probe.py --label padded-pack-fp32-perf --tiles 8192 --fp32-dst --iters 20
```

Each run checks four cases: compact scalar, compact standard width4, padded
scalar, padded custom4. Input is 128 independent random BF16 tiles with explicit
positive/negative tile-ID markers. Every output of the final 128-tile cycle is
downloaded, compared exactly to input, hashed, and saved in a .pt file. Only
correct runs get timing. Both BF16 and FP32 DST are supported.

Maximum CB footprint: BF16 DST 360448 bytes; FP32 DST 311296 bytes.
No alias, matmul consumer, or padding canary is tested. Timing is the entire
copy/pack/unpack/copy/pack roundtrip including configuration and initial/final
128-tile DRAM traffic, not isolated pack speed or attention TFLOPs. It reports
tiles/s, nominal payload GB/s, and cycles/tile at the explicitly assumed clock.

Local verification only: Python syntax/CLI, mocked descriptor configurations,
and an address-counter simulation. No JIT or device calls were run by the author.
