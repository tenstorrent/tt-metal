# DeepSeek-V3 prefill stress + monitor scripts

| File | Purpose |
|---|---|
| `run_iter200_x20.sh` | 20× outer loop: `tt-smi -glx_reset` then iter200 pytest. Each run's stdout is `tee`'d to `<log dir>/log_NN`. No timeout — pytest stays alive on hang for manual debug. |
| `watch_x20_iter200.sh` | Refreshing status table (PASS / HANG? / FAIL / RUN / PENDING) over the run logs. Refresh every 60s. |
| `tail_latest_log_x20.sh` | `tail -10` of the newest `log_NN`, refresh every 30s. |
| `parse_iteration_times.py` | Per-iteration timing extractor (min / avg / max). Reads any pytest log with `Starting iteration:` markers. |

The wrapper writes logs to `/workspace/tt-metal/stress_x20_iter200/` inside the container, which maps to `/home/ubuntu/devs/tt-metal/stress_x20_iter200/` on the host.

---

## Group A — launch from the host (typical)

The stress wrapper runs *inside* `metal-dev` via `docker exec`. The watch/tail monitors run on the *host* against the mounted log directory. All three live in detached `tmux` sessions on the host so they survive disconnects.

```bash
# 1) the stress wrapper (runs inside docker)
tmux new-session -d -s stress20 \
  "docker exec metal-dev bash /workspace/tt-metal/models/demos/deepseek_v3_d_p/scripts/run_iter200_x20.sh \
    |& tee /home/ubuntu/devs/tt-metal/stress_x20_iter200/_master.log"

# 2) status table (runs on host, reads host-side log dir)
tmux new-session -d -s stress20_watch \
  /home/ubuntu/devs/tt-metal/models/demos/deepseek_v3_d_p/scripts/watch_x20_iter200.sh

# 3) tail of latest log (runs on host)
tmux new-session -d -s stress20_tail \
  /home/ubuntu/devs/tt-metal/models/demos/deepseek_v3_d_p/scripts/tail_latest_log_x20.sh

# attach (read-only):
tmux attach -t stress20 -r
tmux attach -t stress20_watch -r
tmux attach -t stress20_tail -r
```

Detach with `Ctrl-b d`. Kill all three: `tmux kill-session -t stress20{,_watch,_tail}`.

---

## Group B — launch from inside the docker container

When you're already inside `metal-dev` (`docker exec -it metal-dev bash`), drop the `docker exec` wrapper. The watch/tail scripts assume the **host** path `/home/ubuntu/devs/tt-metal/stress_x20_iter200/`; from inside the container that path is `/workspace/tt-metal/stress_x20_iter200/`, so they need a one-line override (or symlink the dir to match).

```bash
# 0) (only once) make the host-style path resolve inside docker, so the
#    watch/tail scripts work unmodified:
mkdir -p /home/ubuntu/devs/tt-metal
ln -sfn /workspace/tt-metal/stress_x20_iter200 /home/ubuntu/devs/tt-metal/stress_x20_iter200

# 1) the stress wrapper directly (no docker exec)
tmux new-session -d -s stress20 \
  "bash /workspace/tt-metal/models/demos/deepseek_v3_d_p/scripts/run_iter200_x20.sh \
    |& tee /workspace/tt-metal/stress_x20_iter200/_master.log"

# 2) status table
tmux new-session -d -s stress20_watch \
  /workspace/tt-metal/models/demos/deepseek_v3_d_p/scripts/watch_x20_iter200.sh

# 3) tail of latest log
tmux new-session -d -s stress20_tail \
  /workspace/tt-metal/models/demos/deepseek_v3_d_p/scripts/tail_latest_log_x20.sh

# attach (read-only):
tmux attach -t stress20 -r
tmux attach -t stress20_watch -r
tmux attach -t stress20_tail -r
```

If you'd rather not symlink, edit `DIR=/home/ubuntu/devs/tt-metal/stress_x20_iter200` at the top of `watch_x20_iter200.sh` and `tail_latest_log_x20.sh` to `/workspace/tt-metal/stress_x20_iter200`.

---

## After the run

Per-iteration timing for any single run log:

```bash
python3 models/demos/deepseek_v3_d_p/scripts/parse_iteration_times.py \
  -f /home/ubuntu/devs/tt-metal/stress_x20_iter200/log_05
```

Outputs `min / avg / max` of the inter-iteration period and `sync→next-iteration` window, with outliers (>2× median) filtered out.
