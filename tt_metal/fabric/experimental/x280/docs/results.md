# X280 characterization results

Results for the code in this tree, measured on silicon with the images it builds; each quantity is
defined in [measurement.md](measurement.md).

- A Blackhole p150b, with the X280 at 1750 MHz (`X280_PLL_MHZ`), every member at 14 LIM ways.
- X280 images built by SFPI 7.87.0; Metal's watcher off.
- Benchmark images run from the cached local GDDR, code, data and stacks, with the L2 prefetcher
  on ([device_images.md](device_images.md#linker-script-picks)); pipeline_stage reads its packet
  headers from uncached LIM.
- 3 passing attempts per configuration; 6 for the Metal p2p peers' `*InitiatesFirst` points and
  `packet.latency-*`, 4 for `packet.backpressure-*`.
- Figures are mean [min-max] over attempts (one figure: every attempt agreed), in the tabulator's
  units; core_sync's ticks are raw. At P > 1 an eth_credit rate is per forwarder, over every
  forwarder of every attempt.

## p2p_latency

| member | initiator | peer waits for | RTT wall p50 ns |
|---|---|---|---:|
| x280_same_tile | A | completion | 80.000 |
| x280_same_tile | B | completion | 66.667 [60.000-80.000] |
| x280_peer_tile | A | completion | 200.000 |
| x280_peer_tile | B | completion | 200.000 |
| tensix_peer_tile | X280 | completion | 360.000 |
| tensix_peer_tile | Tensix | completion | 335.309 [334.815-336.296] |
| tensix_peer_tile | X280 | departure | 360.000 |
| tensix_peer_tile | Tensix | departure | 337.284 [337.037-337.778] |
| erisc_peer_tile | X280 | completion | 360.000 |
| erisc_peer_tile | ERISC | completion | 356.667 [332.593-361.481] |
| drisc_peer_tile | X280 | completion | 360.000 |
| drisc_peer_tile | DRISC | completion | 333.210 [332.593-334.074] |

## transport_bandwidth

INJECTION boundary. X280 -> Tensix:

| store B | harts | 64 B run GB/s | 65536 B run GB/s |
|---|---:|---:|---:|
| 4 (`sw`) | 1 | 1.800 | 1.800 |
| 8 (`sd`) | 1 | 3.600 [3.599-3.600] | 3.600 |
| 64 (`vse32.v`) | 1 | 14.392 | 14.400 |
| 4 (`sw`) | 4 |  | 0.412 |
| 8 (`sd`) | 4 |  | 0.824 |
| 64 (`vse32.v`) | 4 |  | 3.294 |

Tensix -> X280:

| 64 B GB/s | 4096 B GB/s | 16384 B GB/s |
|---:|---:|---:|
| 3.755 | 17.893 [17.504-18.671] | 17.503 |

Idle ERISCs -> X280 at 4096 B:

| writers=1 run GB/s | writers=2 run GB/s | writers=4 run GB/s | writers=8 run GB/s |
|---:|---:|---:|---:|
| 17.504 | 17.894 [17.505-18.672] | 17.893 [17.504-18.671] | 17.893 [17.504-18.672] |

## core_sync

Spread of 16 Tensix workers' exit timestamps, by release.

| multicast raw range ticks | unicast raw range ticks |
|---:|---:|
| 126.7 [123-130] | 8956.7 [8943-8964] |

## packet_injection

Tensix BRISC producers, S=16, B=1:

| T | C=1, K=1 ns/header | C=8, K=16 ns/header |
|---:|---:|---:|
| 1 | 696.960 [696.940-696.980] | 695.927 [695.720-696.140] |
| 2 | 347.590 [347.450-347.720] | 347.720 [347.670-347.750] |
| 4 | 173.933 [173.905-173.955] | 174.017 [173.920-174.085] |
| 8 | 104.665 [104.502-104.790] | 87.158 [87.115-87.185] |

One-way latency, one producer:

| header creation | latency p50 ns |
|---|---:|
| included | 1423.210 [1388.889-1465.926] |
| excluded | 1074.938 [1056.296-1100.000] |

Idle-ERISC header ingress, C=1, K=1, B=1:

| writers | ns/header |
|---:|---:|
| 1 | 387.995 [387.949-388.047] |
| 4 | 103.957 [103.877-104.019] |
| 8 | 104.388 [104.341-104.465] |

The backpressure configurations run at P=1, where the credit-stall population is empty by
construction ([packet_injection.md](benchmarks/packet_injection.md#backpressure-measurement)).

## pipeline_stage

`cpu/item` is in X280 `rdcycle` cycles.

| stage | ns/item | cpu/item |
|---|---:|---:|
| route_compute, plain | 44.607 [44.537-44.658] | 78.063 [77.940-78.154] |
| route_compute, instrumented | 62.093 [62.061-62.139] | 108.660 [108.603-108.741] |
| route_forward decision | 24.002 [24.001-24.003] | 42.004 [42.003-42.005] |

## eth_credit

At 4192 B slots, `BATCH` 4:

| stretch | P | Mpkt/s |
|---|---:|---:|
| whole path | 1 | 0.376 |
| whole path | 2 | 0.221 |
| whole path | 4 | 0.180 |
| Ethernet hop alone | 1 | 5.192 [5.179-5.199] |

Credit round trip at one packet in flight:

| loop | slot B | mean ns |
|---|---:|---:|
| hop | 4192 | 1876.121 [1873.104-1877.748] |

The tabulator's ceilings, as it prints them:

| P | N_LINK | N_DELIVER | slot B | BATCH | path Mpkt/s | NoC hop Mpkt/s | W1 Mpkt/s | Ethernet hop Mpkt/s | receive side Mpkt/s | W2 Mpkt/s | R1 Mpkt/s | slowest leg | credit loop Mpkt/s | bound | path/bound |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|---:|---|---:|
| 1 | 8 | 8 | 160 | 1 | 0.356 |  |  | 3.107 |  | 38.043 | 38.043 |  | 4.501 |  |  |
| 1 | 8 | 8 | 160 | 4 | 0.376 | 0.376 | 0.383 | 5.243 | 5.660 | 152.170 | 152.170 | NoC hop | 4.501 | NoC hop | 1.000 |
| 1 | 8 | 8 | 1120 | 1 | 0.356 |  |  | 3.094 |  | 38.043 | 38.043 |  | 4.487 |  |  |
| 1 | 8 | 8 | 1120 | 4 | 0.376 | 0.376 | 0.383 | 5.243 | 5.592 | 152.170 | 152.170 | NoC hop | 4.487 | NoC hop | 1.000 |
| 1 | 8 | 8 | 4192 | 1 | 0.356 |  |  | 3.083 |  | 38.043 | 38.043 |  | 4.264 |  |  |
| 1 | 8 | 8 | 4192 | 4 | 0.376 | 0.376 | 0.383 | 5.192 | 5.650 | 152.170 | 152.170 | NoC hop | 4.264 | NoC hop | 1.000 |
| 1 | 8 | 8 | 16480 | 4 | 0.376 | 0.376 | 0.383 | 2.967 | 5.650 | 152.170 | 152.170 | NoC hop | 3.822 | NoC hop | 1.000 |
| 2 | 8 | 8 | 4192 | 4 | 0.221 | 0.221 | 0.383 |  | 2.857 | 152.170 | 152.170 | NoC hop | 4.264 | NoC hop | 1.000 |
| 4 | 8 | 8 | 4192 | 4 | 0.180 | 0.181 | 0.182 |  | 1.446 | 109.375 | 107.232 | NoC hop | 4.264 | NoC hop | 0.992 |

At P=2 the W1 leg alone lands in one of two states per attempt and slot size, 0.212 or 0.383 Mpkt/s
per forwarder; at 4192 B every attempt here landed on 0.383.

Every other configuration and column is regenerated from the records by the runner and the
tabulator ([measurement.md](measurement.md#collection-and-reproduction)).
