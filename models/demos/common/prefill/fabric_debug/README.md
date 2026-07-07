# Worst-AllGather fabric reproducer

Standalone traced reproducer of the single worst-case CCL op from Kimi-K2.6 prefill, to isolate
the connected-2-galaxy fabric penalty from the full 61-layer model.

## The op
`ttnn.all_gather` (AllGatherDeviceOperation), per-device input `[1,1,640,1792]` bf16 TILE DRAM,
gathered over the 4-device mesh axis -> `[1,1,640,7168]`; `dim=3, cluster_axis=1, num_links=2,
topology=Linear`. Captured into a ttnn trace and replayed 100x with `synchronize_device` before+after
each replay; reports host latency mean/min/max/median and (via `TT_METAL_DEVICE_PROFILER=1`) the
per-op DEVICE KERNEL DURATION in the ops CSV.

## Run
    # single galaxy (this host only, no peer)
    bash run_worst_ag_single.sh 1d      # FABRIC_1D
    bash run_worst_ag_single.sh 2d      # FABRIC_2D
    # connected 2-galaxy (this host = rank0 runs the AllGather; peer d09u02 idles holding the fabric)
    bash run_worst_ag_pipe.sh
    # all three, prints device-kernel mean/min/max per leg -> logs/worst_ag_summary.txt
    bash run_worst_ag_all.sh

Profiler CSVs land in /data/ppopovic/prof_out/worst_ag_*; logs in ./logs/.
Connected leg uses ttrun + bindings/worst_ag_2galaxy.yaml (connected MGD) and needs working SSH to
the peer (host d09u08=rank0/local, d09u02=rank1/peer; edit LOCAL/PEER in run_worst_ag_pipe.sh).

## Baseline (AllGather device kernel duration, 100 replays)
| setup            | fabric     | mesh       | mean    | min     | max     |
|------------------|------------|------------|---------|---------|---------|
| single_1D        | FABRIC_1D  | 1 galaxy   | ~178 us | ~175 us | ~212 us |
| single_2D        | FABRIC_2D  | 1 galaxy   | ~182 us | ~179 us | ~214 us |
| connected_2D     | FABRIC_2D  | 2 galaxy   | ~285 us | ~282 us | ~318 us |

=> 1D->2D single-galaxy is free (~1.02x); single-2D -> connected-2D is ~1.57x, entirely fabric
(the same AllGather runs on one galaxy's 32 chips; no data crosses the inter-galaxy link).
