# X280 characterization results

Results for the code in this tree, measured on silicon with the images it builds; each quantity is
defined in [measurement.md](measurement.md).

- A Blackhole p150b, with the X280 at 1750 MHz (`X280_PLL_MHZ`), every member at 14 LIM ways.
- X280 images built by SFPI 7.87.0; Metal's watcher off.
- 3 passing attempts per configuration; 6 for the Metal p2p peers' `*InitiatesFirst` points and
  `packet.latency-*`, 4 for `packet.backpressure-*`.
- Figures are mean [min-max] over attempts (one figure: every attempt agreed), in the tabulator's
  units; core_sync's ticks are raw. At P > 1 an eth_credit rate is per forwarder, over every
  forwarder of every attempt.

## p2p_latency

| member | initiator | peer waits for | RTT wall p50 ns |
|---|---|---|---:|
| x280_same_tile | A | completion | 100.000 |
| x280_same_tile | B | completion | 100.000 |
| x280_peer_tile | A | completion | 240.000 |
| x280_peer_tile | B | completion | 240.000 |
| tensix_peer_tile | X280 | completion | 346.667 [340.000-360.000] |
| tensix_peer_tile | Tensix | completion | 373.827 [373.333-374.074] |
| tensix_peer_tile | X280 | departure | 360.000 |
| tensix_peer_tile | Tensix | departure | 368.889 |
| erisc_peer_tile | X280 | completion | 360.000 |
| erisc_peer_tile | ERISC | completion | 361.481 |
| drisc_peer_tile | X280 | completion | 356.667 [340.000-360.000] |
| drisc_peer_tile | DRISC | completion | 368.889 |

## transport_bandwidth

INJECTION boundary. X280 -> Tensix:

| store B | harts | 64 B run GB/s | 65536 B run GB/s |
|---|---:|---:|---:|
| 4 (`sw`) | 1 | 1.800 | 1.800 |
| 8 (`sd`) | 1 | 3.599 | 3.600 |
| 64 (`vse32.v`) | 1 | 14.386 [14.384-14.388] | 14.399 |
| 4 (`sw`) | 4 |  | 0.412 |
| 8 (`sd`) | 4 |  | 0.824 |
| 64 (`vse32.v`) | 4 |  | 3.294 |

Tensix -> X280:

| 64 B GB/s | 4096 B GB/s | 16384 B GB/s |
|---:|---:|---:|
| 3.755 | 17.504 | 17.503 |

Idle ERISCs -> X280 at 4096 B:

| writers=1 run GB/s | writers=2 run GB/s | writers=4 run GB/s | writers=8 run GB/s |
|---:|---:|---:|---:|
| 17.504 | 17.894 [17.504-18.672] | 17.504 | 17.504 |

## core_sync

Spread of 16 Tensix workers' exit timestamps, by release.

| multicast raw range ticks | unicast raw range ticks |
|---:|---:|
| 119.7 [116-123] | 11559.3 [11544-11568] |

## packet_injection

Tensix BRISC producers, S=16, B=1:

| T | C=1, K=1 ns/header | C=8, K=16 ns/header |
|---:|---:|---:|
| 1 | 696.273 [696.220-696.340] | 694.753 [694.640-694.920] |
| 2 | 349.043 [348.860-349.190] | 348.490 [348.350-348.670] |
| 4 | 234.095 [233.880-234.215] | 174.415 [174.350-174.520] |
| 8 | 230.629 [230.615-230.640] | 87.394 [87.373-87.435] |

One-way latency, one producer:

| header creation | latency p50 ns |
|---|---:|
| included | 1984.321 [1974.815-2003.704] |
| excluded | 1637.531 [1625.185-1644.444] |

Idle-ERISC header ingress, C=1, K=1, B=1:

| writers | ns/header |
|---:|---:|
| 1 | 388.079 [388.066-388.105] |
| 4 | 232.653 [232.617-232.686] |
| 8 | 229.575 [229.536-229.622] |

The backpressure configurations run at P=1, where the credit-stall population is empty by
construction ([packet_injection.md](benchmarks/packet_injection.md#backpressure-measurement)).

## pipeline_stage

`cpu/item` is in X280 `rdcycle` cycles.

| stage | ns/item | cpu/item |
|---|---:|---:|
| route_compute, plain | 41.824 [41.708-41.884] | 73.190 [72.987-73.294] |
| route_compute, instrumented | 72.184 [72.129-72.236] | 126.322 [126.228-126.414] |
| route_forward decision | 25.148 | 44.007 [44.007-44.008] |

## eth_credit

At 4192 B slots, `BATCH` 4:

| stretch | P | Mpkt/s |
|---|---:|---:|
| whole path | 1 | 0.372 |
| whole path | 2 | 0.215 |
| whole path | 4 | 0.177 |
| Ethernet hop alone | 1 | 5.180 [5.171-5.195] |

Credit round trip at one packet in flight:

| loop | slot B | mean ns |
|---|---:|---:|
| hop | 4192 | 1930.818 [1929.934-1932.244] |

The tabulator's ceilings, as it prints them:

| P | N_LINK | N_DELIVER | slot B | BATCH | path Mpkt/s | NoC hop Mpkt/s | W1 Mpkt/s | Ethernet hop Mpkt/s | receive side Mpkt/s | W2 Mpkt/s | R1 Mpkt/s | slowest leg | credit loop Mpkt/s | bound | path/bound |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|---:|---|---:|
| 1 | 8 | 8 | 160 | 1 | 0.342 |  |  | 3.107 |  | 38.043 | 38.043 |  | 4.350 |  |  |
| 1 | 8 | 8 | 160 | 4 | 0.372 | 0.371 | 0.383 | 5.243 | 4.594 | 152.170 | 152.170 | NoC hop | 4.350 | NoC hop | 1.000 |
| 1 | 8 | 8 | 1120 | 1 | 0.342 |  |  | 3.095 |  | 38.043 | 38.043 |  | 4.356 |  |  |
| 1 | 8 | 8 | 1120 | 4 | 0.372 | 0.371 | 0.383 | 5.245 | 4.550 | 152.170 | 152.170 | NoC hop | 4.356 | NoC hop | 1.000 |
| 1 | 8 | 8 | 4192 | 1 | 0.342 |  |  | 3.083 |  | 38.043 | 38.043 |  | 4.143 |  |  |
| 1 | 8 | 8 | 4192 | 4 | 0.372 | 0.371 | 0.383 | 5.180 | 4.595 | 152.170 | 152.170 | NoC hop | 4.143 | NoC hop | 1.000 |
| 1 | 8 | 8 | 16480 | 4 | 0.371 | 0.371 | 0.383 | 2.967 | 4.596 | 152.170 | 152.170 | NoC hop | 3.715 | NoC hop | 1.000 |
| 2 | 8 | 8 | 4192 | 4 | 0.215 | 0.217 | 0.326 |  | 2.855 | 152.170 | 152.170 | NoC hop | 4.143 | NoC hop | 0.989 |
| 4 | 8 | 8 | 4192 | 4 | 0.177 | 0.180 | 0.182 |  | 1.456 | 107.237 | 109.381 | NoC hop | 4.143 | NoC hop | 0.981 |

Every other configuration and column is regenerated from the records by the runner and the
tabulator ([measurement.md](measurement.md#collection-and-reproduction)).
