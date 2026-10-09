# Prepared decoder precision controls

These configurations are prepared and validated by the policy loader, but have
**not run on hardware, been queued, or changed the serving default**. They are
controls for the numerical investigation after the queued HF layer comparison.
There is no accuracy or throughput claim for either policy.

Both retain the BFP8/HiFi2 head, FP32 recurrent state, BFP8 KV, BF16 activations,
native recurrence and accurate-full-tile decode attention of the current head
control. The differences are explicit:

| Configuration | Decoder weights | Decoder projection fidelity | Purpose |
|---|---|---|---|
| `precision_accurate_decode_bfp8_head.json` | BFP4 | LoFi | Existing head control |
| `precision_accurate_decode_hifi2.json` | BFP4 | HiFi2 | Isolate projection arithmetic fidelity |
| `precision_accurate_decode_bfp8_all.json` | BFP8 | HiFi2 | Then isolate decoder weight quantization |

The effective policies were loaded with `tt/precision.py::load_precision` and
checked through `decoder_policy` for all 64 layers. The validation receipt
records their effective-policy fingerprints, source hash and exact differences.
This validates configuration interpretation, not numerical execution.

## Static capacity check

`checkpoint-shapes.json` records read-only checkpoint configuration and relevant
safetensor header shapes with hashes. `capacity-estimate.json` models one TP4
chip with 16 slots and the unchanged 1,050,592-token KV pool per replica. It
includes both retained interleaved and DRAM-sharded projection weights, with
the current reader-dependent padding, plus the head, embedding, rotary tables,
KV, base recurrent/conv state, a maximum-size resident packed state and the
200,000,000-byte trace reservation.

For each matrix tuple `(layers, K, local_N, readers)`, DRAM columns are rounded
up to a multiple of `8 * readers * 32`. Interleaved and padded tile counts use
32x32 tiles at 576 bytes for BFP4 or 1,088 bytes for BFP8. Head vocabulary shards
are split into three 16,384-column pieces and a padded 13,312-column tail.

| Decoder policy | Known resident sum/chip | Remaining versus SoC bank map |
|---|---:|---:|
| BFP4, BFP8 head | 17.669 GiB | 14.206 GiB |
| BFP8, BFP8 head | 23.477 GiB | 8.398 GiB |

The decoder weight increase is approximately **5.808 GiB/chip**. The SoC bank
map totals 31.875 GiB/chip. This is a static estimate, **not live free memory or
proof that model startup fits**: norms/masks/constants, CCL workspace, temporary
prefill/matmul/sampling/state-packing buffers, allocator alignment and
fragmentation, and other runtime reservations are excluded. Hardware admission
and peak-memory checks are required before a full-model run.

Higher weight precision increases memory traffic; HiFi2 may increase arithmetic
cost. Those costs have not been measured here. These controls must pass the
same full GPQA protocol and receive measured performance results before any
promotion. Changing precision or retaining the same sampling seed does not
guarantee identical stochastic trajectories or establish a kernel bug.
