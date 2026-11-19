# Affine/gate V3: direct face reads, NoC staging copies and gamma broadcast

Source-only follow-up to `gdn_affine_gate_flat_v2.py`. Parent evidence says V2
passed its 16 real-gamma exactness cases, with about 49 us helper time versus
55 us native chain and only about 7 us whole-path gain at 64 cores. The present
candidate is unmeasured and makes no device-equivalence or speed claim.

Native RMSNorm stays outside the helper. The intended arithmetic remains:

1. Multiply BF16 normalized heads by BF16 gamma in FP32.
2. Explicitly round that affine result to BF16.
3. Evaluate native FP32 SiLU on the BF16 gate widened to FP32.
4. Multiply by the rounded affine result, then explicitly round to BF16.

The output remains compact `[1,B,6144]` BF16 TILE in DRAM. Supplied output
allocation, program-cache/trace arguments, existing output writer and all
padding/shape support remain intact.

## Reader audit and replacement

V2 gathers BF16 normalized face rows through aligned64-byte scratch, then
executes eight RISC word copies for every user/face into the output norm tile.
It also replicates gamma across every user row with another eight word copies
per user/face. At B32 this is 512 norm plus 512 gamma word stores per output
tile, or 196,608 stores across the 192 tiles. Its padded gate rows are cleared
by further RISC stores for smaller batches.

V3 replaces these loops:

- Source head row is `(user*48+head)%32`. Its parity is constant for a given
  head, while output user-row parity alternates. When source and destination
  byte addresses agree modulo64, a direct32-byte DRAM read fills that face row.
  Otherwise a64-byte aligned pair is staged and a local32-byte NoC `tt_memmove`
  copies the desired half. The comparison includes actual CB base address;
  it does not assume a favorable allocation alignment.
- Gamma's two face rows are still read as64-byte aligned chunks, then copied
  with two local32-byte NoC operations into row0 of its CB. Compute uses
  `sfpu_mul_bcast_row<true>(0,1)` rather than loading an explicitly replicated
  gamma tile. The row-broadcast API documents that only source row0 is used;
  other gamma CB rows are irrelevant.
- Invalid user rows in norm and gate are zeroed in contiguous face segments
  through `async_write_zeros`. On Blackhole these use local read-loopback, and
  `write_zeros_l1_barrier` completes both staging copies and pad zeros before
  any CB is published.

At the usual64-byte-aligned CB base, half of the B32 norm face reads bypass
staging. Norm DRAM reads fall from4,096 to3,072 bytes/output tile, saving
196,608 bytes per invocation. The meaningful opportunity is removal of RISC
element-copy instructions, not a large DRAM-bandwidth reduction. Local NoC
copies, an additional barrier and SFPU gamma-broadcast work are costs that can
negate the gain. Only measured whole-path latency selects the candidate.

Source support:

- `prefill_unflatten_bf16_v2.py` already uses parity-matched32-byte DRAM face
  reads; a small transfer is legal when the source/destination modulo64 agrees.
- `tt_metal/hw/inc/api/compute/sfpu_binary_bcast.h:287-375` specifies FP32
  row-broadcast multiplication and meaningful values only in source row0.
- `tt_metal/hw/inc/internal/tt-1xx/noc_zero_l1.inl` implements local zero as
  NoC read-loopback, with a read barrier completing it.
- The local `tt_memmove<false,true,true,32>` path is also used by the native
  QKV splitter for aligned BF16 face-row copies.

## Immutable artifacts and checks

- Helper: `candidates/gdn_affine_gate_flat_v3.py`.
- Frozen decoder: `candidates/gdn_affine_gate_v3_on162.py`.
- Generator: `candidates/make_gdn_affine_gate_decode_v3.py`.
- CPU checker/results: `candidates/check_gdn_affine_gate_v3_cpu.py` and
  `candidates/gdn_affine_gate_v3_cpu.json`.

The generated decoder enables the helper by default only in decode;
`gdn_fused_affine_gate=False` is the matched native-chain control. Set
`gdn_affine_gate_cores=64` to reproduce the earlier measured worker count.
Prefill, recurrence, native RMSNorm and output projection remain unchanged.

Python syntax and CPU/source checks passed. The checker verifies gathered
16-bit elements and padding at B1/3/31/32 for CB base offsets0/16/32/48 modulo64.
Writer source is byte-for-byte unchanged. Compute source differs only by the
gamma row-broadcast include/call; both BF16 conversions and FP32 SiLU are
unchanged. CPU mapping checks do not emulate the SFPU broadcast arithmetic.

## Parent device queue

The existing real-gamma exactness probe is API-compatible:

```bash
python_env/bin/python -m models.autoports.qwen_qwen3_6_27b.tests.run_gdn_affine_gate_probe \
  --candidate models/autoports/qwen_qwen3_6_27b/doc/optimized_decoder/candidates/gdn_affine_gate_flat_v3.py \
  --cores 64 \
  --output models/autoports/qwen_qwen3_6_27b/doc/optimized_decoder/affine_gate_v3_probe.json
```

Require all 16 cases to remain bit-exact against the unchanged TTNN chain,
including repeated supplied output and saturated/near-zero gates. Then measure
the frozen whole decoder with the same64-core policy, short/512-step gates,
traced replay and watcher. Retain only a consistent whole-path improvement.
No hardware was used by this source-only helper.
