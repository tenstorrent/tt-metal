# Frozen profiler accounting

These tables are generated from the frozen-source enriched CSVs under
`tracy/current_fused_final/`. Blackhole aggregate DRAM bandwidth is 512 GB/s.
For each window, modeled DRAM bytes are reconstructed from each row's reported
DRAM GB/s times device duration. The theoretical roofline time is
`modeled_bytes / 512e9`; efficiency is `roofline_time / summed_device_time`.
Device+gap and host come from the same signposted run.

| Window | Host ms | Device ms | Gap ms | Device+gap ms | Modeled bytes | Roofline ms | Efficiency |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| sliding decode | 1.406574 | 1.305596 | 0.076782 | 1.382378 | 157171712 | 0.306976 | 23.51% |
| full decode | 1.556952 | 1.460519 | 0.071672 | 1.532191 | 191774720 | 0.374560 | 25.65% |
| sliding prefill seq256 | 21.559562 | 21.160643 | 0.298680 | 21.459323 | 6400770048 | 12.501504 | 59.08% |
| full prefill seq256 | 21.746347 | 21.526982 | 0.116447 | 21.643429 | 6438518784 | 12.575232 | 58.42% |

Host minus device+gap is 0.024196/0.024761 ms for decode and
0.100239/0.102918 ms for prefill. The corresponding `*_report.csv` files are
the readable per-operation tables; `*_stacked.csv` and PNG files aggregate the
same rows by operation.

## Required tilize/untilize boundaries

| Window | Operation | Count | Total us | Required boundary |
| --- | --- | ---: | ---: | --- |
| sliding decode | TilizeWithValPadding | 1 | 5.653 | row-major sparse routing input to tiled sparse matmul |
| sliding decode | UntilizeWithUnpadding | 4 | 7.036 | tiled router values/indices and sparse outputs to row-major TopK/scatter/selection consumers |
| full decode | TilizeWithValPadding | 1 | 5.670 | same router-to-sparse contract |
| full decode | UntilizeWithUnpadding | 4 | 7.042 | same TopK/scatter/selection contracts |
| sliding prefill | Tilize | 1 | 2.973 | routing weights into tiled expert reduction |
| sliding prefill | Untilize | 1 | 4.787 | router TopK/scatter boundary |
| sliding prefill | UntilizeWithUnpadding | 2 | 20.929 | packed 704/704 projection slices and routing consumer contract |
| full prefill | Tilize | 1 | 2.947 | routing weights into tiled expert reduction |
| full prefill | Untilize | 1 | 4.802 | router TopK/scatter boundary |
| full prefill | UntilizeWithUnpadding | 2 | 20.962 | packed projection slices and routing consumer contract |

No conversion crosses the host boundary: there are no runtime Torch,
`from_torch`, `to_torch`, or host fallback rows. The retained conversions are
device-side layout contracts, total 0.97% of sliding prefill device time and
0.95% of full prefill device time.
