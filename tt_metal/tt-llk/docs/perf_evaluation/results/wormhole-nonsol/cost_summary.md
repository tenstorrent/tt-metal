# Perf gate budget -- wormhole, non speed-of-light

markers: `perf and not accuracy`
compile -n 10, measure -n 15, cold build per config: yes

| config | run types | compile | measure | total | rows | points | modules | rc c/m |
|---|---|--:|--:|--:|--:|--:|--:|--:|
| full | all declared | 0:21:57 | 0:08:17 | 0:30:14 | 0 | 0 | 0 | 0/0 |
| isolates | UNPACK_ISOLATE,MATH_ISOLATE,PACK_ISOLATE | 0:13:07 | 0:05:38 | 0:18:45 | 0 | 0 | 0 | 0/0 |
| l1 | L1_TO_L1 | 0:06:03 | 0:03:16 | 0:09:19 | 0 | 0 | 0 | 0/0 |

`rc c/m` is the compile/measure pytest exit code. A non-zero measure code
means the sweep did not finish, so that row's time is a floor, not a cost.
