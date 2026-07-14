# Conv Pipeline Watcher

Sibling of the SDPA pipeline watcher (`../.sdpa-watch/`) — **same infrastructure**,
different scope and channel. Monitors `tenstorrent/tt-metal` GitHub Actions
workflows on `main` for failures in the ops/models **owned by
`@tenstorrent/metalium-developers-convolutions`** (per `.github/CODEOWNERS`) and
posts a digest to a dedicated Slack channel (`#conv-watch`).

Scope = the conv family (conv2d/1d/3d, conv_transpose2d, maxpool/avgpool/global/
adaptive pool, upsample, grid_sample, fold, sliding_window, cnn/tt_cnn) plus the
two conv-team-owned models **resnet50** and **stable diffusion 1.4**.

## Runtime vs snapshot (read `../.sdpa-watch/README.md` + `SETUP.md` first)

- **Runtime** = `~/.conv-watch/` — the cron job, cache, secrets, logs. Source of truth.
- **Repo snapshot** = this dir — a secrets-free copy of the functional files for
  review/porting. Only `config.sh`, `watch.sh`, `agent_prompt.txt`,
  `ensure-cron.sh` are tracked; `slack_webhook`, `oauth_token`, `state.json`,
  and `*.log` are runtime-only and never committed.

## Differences from the SDPA watcher

- `watch.sh` is **self-locating** (`SDPA_HOME` derived from `BASH_SOURCE`), so the
  dir is a clean copy — it reads its own `~/.conv-watch/` config/state, never sdpa's.
- Digest title is "Conv Pipelines"; `agent_prompt.txt` scopes triage to conv/pool.
- Cron fires at **:30** (sdpa is :00) to avoid the shared `~/.claude/.credentials.json`
  OAuth-refresh race. `ensure-cron.sh` MARKER/CRON_LINE point at `.conv-watch`.
- Watches **15 pipelines** (see `config.sh` `PIPELINES`): 3 sanity gates (Sanity,
  Blackhole Sanity, Debug Sanity), Merge Gate C++ smoke, L2 Nightly conv/pool +
  tt-cnn + di/dt, Device Perf, Frequent Models (resnet50 + SD1.4 PCC), Single-card
  Demos, Model Perf, 4× T3K, Galaxy, and conv/pool Sweeps.

## Setup on a new host

Follow `../.sdpa-watch/SETUP.md`, but: copy these files into `~/.conv-watch/`,
create a `#conv-watch` Slack incoming webhook → `~/.conv-watch/slack_webhook`,
add the `~/.bashrc` hook for `~/.conv-watch/ensure-cron.sh`, and install the
`30 * * * *` crontab line. Auth (`gh`, Claude credential) is shared with the
sdpa watcher — nothing extra to configure.
