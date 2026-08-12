"""Where do the rows live? Sizes the two levers before either costs device time.

Item 1 measured ~670 ms of the 930 ms decode as data-proportional, and the recorded fact is that a row
costs ~4.2 ns almost regardless of width. So counting rows per stage is a first-order cost model, and it
decides which lever is worth the work:

  * ``ups`` (ConvTranspose1dViaConv3d) does NOT shard -- audio_ops.py:1678 says "Inner conv stays
    UNSHARDED; forward gathers T, runs unsharded, then re-partitions". Under T-sharding every chip
    redoes these in full and pays an all-gather + re-partition + 4 layout conversions per stage.
  * the AMP resblocks DO shard, via the ``_t_neighbor_pad`` halo.

If the ups hold most of the rows, T-sharding is structurally capped and fixing the ups is item 2. If the
resblocks hold most of the rows, sharding already covers the expensive part and the 8-chip result being
0.898 s (1.04x) must be overhead, not replicated work -- a different fix.
"""

RATES = [5, 5, 2, 2, 2, 2, 2]
KERNELS = [9, 9, 4, 4, 4, 4, 4]
RESBLOCK_KS = [3, 7, 11]  # num_kernels = 3 parallel AMP branches
DILATIONS = [1, 3, 5]
DECODER_DIM = 1024
T_LATENT = 207

print(f"rates {RATES}  total upsample {eval('*'.join(map(str, RATES)))}x")
print(f"{T_LATENT} latents -> {T_LATENT * 800} samples\n")
print(f"{'stage':>5} {'C_in':>5} {'C_out':>6} {'T_out':>8} {'ups rows':>10} {'resblock rows':>14}")
print("-" * 56)

T = T_LATENT
ups_rows = 0
res_rows = 0
per_stage = []
for i, (rate, _k) in enumerate(zip(RATES, KERNELS)):
    c_in = DECODER_DIM // (2**i)
    c_out = DECODER_DIM // (2 ** (i + 1))
    T_out = T * rate

    # The ups conv runs on the zero-stuffed sequence, i.e. T_out rows, once.
    u = T_out

    # Each of the 3 AMP branches has, per dilation, conv1 + conv2 and two snake activations.
    # Every one of those is a full pass over T_out rows.
    convs_per_branch = 2 * len(DILATIONS)
    snakes_per_branch = 2 * len(DILATIONS)
    r = len(RESBLOCK_KS) * (convs_per_branch + snakes_per_branch) * T_out

    ups_rows += u
    res_rows += r
    per_stage.append((i, c_in, c_out, T_out, u, r))
    print(f"{i:>5} {c_in:>5} {c_out:>6} {T_out:>8} {u:>10,} {r:>14,}")
    T = T_out

total = ups_rows + res_rows
print("-" * 56)
print(f"{'':>5} {'':>5} {'':>6} {'':>8} {ups_rows:>10,} {res_rows:>14,}")
print(f"\nups rows        {ups_rows:>12,}  {100 * ups_rows / total:5.1f}%  (NOT sharded)")
print(f"resblock rows   {res_rows:>12,}  {100 * res_rows / total:5.1f}%  (sharded via halo)")
print(f"ratio resblock:ups = {res_rows / ups_rows:.1f} : 1")

print("\nrow-work model under T-sharding (ups replicated, resblocks divided):")
DATA_MS = 670.0  # item 1's data-proportional component at T=207
FLOOR_MS = 260.0  # item 1's T-independent component
ups_share = ups_rows / total
for factor in (1, 4, 8, 32):
    data = DATA_MS * (ups_share + (1 - ups_share) / factor)
    print(
        f"  factor {factor:>2}: floor {FLOOR_MS:5.0f} + data {data:6.1f} = {FLOOR_MS + data:6.1f} ms "
        f"(before any CCL / layout-conversion cost)"
    )

print("\nMeasured for comparison: factor=8 records 0.898 s (test_audio_minimax_h3.py:723).")
print("The gap between that and the row model above is sharding overhead, not replicated work.")

print("\ntail stages are where C is small -- item 3's channel multiplier targets these:")
for i, c_in, c_out, T_out, _u, _r in per_stage[-3:]:
    print(f"  stage {i}: C {c_in} -> {c_out}, T_out {T_out:,}")
