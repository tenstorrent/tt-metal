# Perf gate budget -- blackhole, non speed-of-light

markers: `perf and not accuracy`
compile -n 10, measure -n 15, cold build per config: yes

| config | run types | compile | measure | total | rc c/m |
|---|---|--:|--:|--:|--:|
| full | all declared | 0:15:33 | 0:04:08 | 0:19:41 | 0/0 |
| isolates | UNPACK_ISOLATE,MATH_ISOLATE,PACK_ISOLATE | 0:09:55 | 0:02:45 | 0:12:40 | 0/0 |
| l1 | L1_TO_L1 | 0:04:21 | 0:01:56 | 0:06:17 | 0/0 |

Yield columns omitted: the script counts them in the wrong directory.
Timings are unaffected.
