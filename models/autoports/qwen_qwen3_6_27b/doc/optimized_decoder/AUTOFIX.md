# AutoFix: shared DRAM weight copy

Failing contract: BFP8 decoder prefill/decode must preserve PCC >=.995 and state.
The first two-reader integration produces prefill .576 and recurrent state near zero.
The same BFP8 weights in interleaved storage pass. Legal isolated decode matmuls pass.

Hypotheses and discriminating checks:
1. 2D prefill derives bank strides from logical N while DRAM shard storage rounds N
   to banks*readers*32. Compare first projections with and without explicit zero padding.
2. DRAM conversion corrupts weights. Read back quantized weights and use them as
   the torch matmul control; compare aligned-width siblings.
3. GDN recurrence breaks under new precision. Check projection before recurrence.

`run_shard_probe.py` proves (1): packed GDN N16512 has physical N16896 and PCC .028991;
explicit pad restores .999994543. All aligned siblings already pass .999994 or better.
This occurs before recurrence and with identical quantized weights, excluding (2)/(3).
Source corroboration: matmul_multicore_reuse_mcast_2d_program_factory.cpp computes
per_core_N_storage=ceil(N/num_dram_banks), ignoring wider stored shards. The native
DRAM-sharded decode kernel uses the actual shard width, explaining its passing probe.
Independent AutoDebug launched with the installed script. Initial bwrap namespace
preflight failed; retried with AUTODEBUG_SKIP_CHILD_SANDBOX=1 under the existing
unrestricted execution authorization. The child remains inspection-only.

Minimal proposed repair: zero-pad each dense weight's logical N at setup to its
physical per-bank extent, retain original N, and slice the matmul output before any
consumer. One DRAM copy serves both phases. No shared-kernel edits or context reduction.
Integrated repair validation pending.

Repair1 validated: dram2_padfix_0.json passes prefill/decode .999908/.999989;
1.351ms traced decode, 4.150ms warmed prefill. Pipeline variant passes at1.125ms decode.
Independent AUTODEBUG.md confirms the exact 66-versus-65-tile DRAM stride defect.

Repair2: reducing 2D out_block_w below per_core_N produced NaNs in packed MLP and
batch32 prefill. Independent inspection AUTODEBUG_blocks.md proves the width-sharded
B reader reads whole-worker spans into partial-width CBs. Kept full-width output
blocks and bounded internal matrix rows (512 ordinary,256 packed wide projection,
K2 for wide projection). Slice endpoints explicitly clamp to the remaining logical
rows, since TTNN slice does not apply Python's implicit end clamping.
rowchunks_tailfix_0_b32.json passes batch32 prefill/decode/state checks. This repair
changes no public alignment or context requirement. Packed-MLP repair validation follows.
