# Local owned-process closure

At 2026-09-28 12:17 UTC, selected API PID 38585 received SIGINT after the final
suite/replay completed. Its launcher exited zero; EngineCore PID 38641 exited.
The complete server log was copied to `acceptance_sync/server.log` before the
537-artifact snapshot. `final_server_cleanup.log` records no owner on device
nodes 0–3 and no listener on port 8000.

After repository formatting finished, `docker stop --time 30 9d99fc35c498`
stopped only the owned holder container `intelligent_moser`. Its original
`sleep infinity` session exited 143 as expected from termination. At 12:19:40 UTC,
the follow-up device/port check remained empty. `docker ps` listed only the
two pre-existing foreign containers, both untouched:

```text
853faf68e2b0 mvasiljevic-ttxla Up 3 weeks
0a3845649cf4 tt-xla-ird-svuckovic Up 3 weeks
State Recv-Q Send-Q Local Address:Port Peer Address:PortProcess
```

Checks used `fuser -v /dev/tenstorrent/0 /dev/tenstorrent/1
/dev/tenstorrent/2 /dev/tenstorrent/3` and `ss -ltnp 'sport = :8000'`.
No reset or foreign-process stop was needed for final cleanup. The model's
inherited nanobind teardown warnings remain in the logs; actual server exit
and device-owner checks, not warning suppression, establish cleanup.

The holder was auto-removed by its wrapper. No model artifacts, operator files,
foreign containers, or caches were deleted. Repository checks now use the
owned host test environment, so no device-owning container needs restarting.
